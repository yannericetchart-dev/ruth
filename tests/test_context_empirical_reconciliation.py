"""No-GPU reconciliation of saved SIGABRT context boundaries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibration.gpu_context import reconcile_empirical_context_from_previous_report
from calibration.profile_pipeline import Journal, atomic_json


def success_probe(n_ctx):
    return {
        "requested_n_ctx": n_ctx, "success": True,
        "measurements": {"vram_peak_sampled_mib": 6139.3},
        "full_gpu_native_verification": {"verified": True},
        "gpu_runtime_validation": {
            "verified": True,
            "kv": {"verified_gpu_resident": True},
        },
    }


def blocked_report():
    return {
        "phase": "full_gpu_resident_reference_context",
        "model_id": "qwen-2.5-7b-instruct", "filename": "Q5_K_L.gguf",
        "quantization": "Q5_K_L", "gpu_total_mib": 6141.0,
        "success": False, "measurement_completed": False,
        "confirmation_completed": True, "maximality_proven": False,
        "confirmed_max_gpu_resident_n_ctx": None,
        "confirmed_gpu_resident_n_ctx": 2048,
        "candidate_max_gpu_resident_n_ctx": 2048,
        "highest_gpu_resident_n_ctx_observed": 2048,
        "highest_context_tested_and_failed": 2049,
        "search_ceiling": 32768,
        "uncertain_failure_boundaries": [
            {"n_ctx": 2049, "signal": "SIGABRT",
             "failure_class": "worker_protocol_error"},
            {"n_ctx": 2304, "signal": "SIGABRT",
             "failure_class": "worker_protocol_error"},
        ],
        "confirmation": [
            {"success": True, "vram_peak_sampled_mib": 6139.3},
            {"success": True, "vram_peak_sampled_mib": 6139.3},
        ],
        "attempts": [
            success_probe(2048), success_probe(2048), success_probe(2048),
            {
                "requested_n_ctx": 2049, "success": False,
                "failure_class": "worker_protocol_error",
                "worker_returncode": -6, "worker_signal": "SIGABRT",
            },
        ],
        "validated_gpu_resident_context": {
            "n_ctx": 2048, "native_cuda_validation": True,
            "confirmation_rounds_passed": 2,
            "maximality_proven": False,
            "operationally_safe": False,
        },
        "error": "Legacy uncertain SIGABRT",
    }


class LegacyEmpiricalReportTests(unittest.TestCase):
    def test_adjacent_sigabrt_promoted_to_empirical_max_not_oom(self):
        old = blocked_report()
        new = reconcile_empirical_context_from_previous_report(old)
        self.assertIsNotNone(new)
        self.assertTrue(new["success"])
        self.assertTrue(new["measurement_completed"])
        self.assertEqual(new["confirmed_max_gpu_resident_n_ctx"], 2048)
        self.assertEqual(new["boundary_n_ctx"], 2049)
        self.assertEqual(new["maximality_basis"], "empirical_fixed_runtime_search")
        self.assertTrue(new["failure_cause_unresolved"])
        self.assertEqual(new["uncertain_failure_boundaries"],
                         old["uncertain_failure_boundaries"])
        self.assertFalse(old["success"])

    def test_never_upgrade_without_adjacent_failure_or_full_cuda_proofs(self):
        for variant in ("gap", "no_sigabrt", "kv_cpu", "not_repeated",
                        "successful_higher", "other_boundary_cause"):
            with self.subTest(variant=variant):
                value = blocked_report()
                if variant == "gap":
                    value["highest_context_tested_and_failed"] = 2050
                if variant == "no_sigabrt":
                    value["attempts"][-1]["worker_signal"] = "SIGKILL"
                if variant == "kv_cpu":
                    for row in value["attempts"][:3]:
                        row["gpu_runtime_validation"]["kv"][
                            "verified_gpu_resident"
                        ] = False
                if variant == "not_repeated":
                    value["confirmation"] = [{"success": True}]
                if variant == "successful_higher":
                    value["attempts"].append(success_probe(2050))
                if variant == "other_boundary_cause":
                    value["uncertain_failure_boundaries"][0]["signal"] = "SIGKILL"
                self.assertIsNone(
                    reconcile_empirical_context_from_previous_report(value)
                )

    def test_journal_reconciles_cached_run_without_any_native_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "run"
            folder.mkdir()
            weights = root / "Q5_K_L.gguf"
            weights.write_bytes(b"fake")
            machine = {
                "profile_key": "run",
                "hardware": {"vram_total_mib": 6141.0},
                "choices": {"model_id": "qwen-2.5-7b-instruct",
                            "source": "bartowski"},
                "reference_runtime": {"context_confirm_rounds": 2},
            }
            journal = Journal(folder, machine)
            atomic_json(folder / "baseline.json", {
                "success": True, "baseline": {
                    "filename": weights.name,
                    "local_path": str(weights),
                    "quantization": "Q5_K_L",
                    "native_verified_all_gpu": True,
                }
            })
            journal.phase_state("baseline", "completed", "baseline.json")
            original = blocked_report()
            atomic_json(folder / "context_gpu.json", original)
            journal.phase_state(
                "context_gpu", "blocked", "context_gpu.json",
            )
            with patch("calibration.gpu_context.probe_context") as probe:
                journal.reconcile_blocked_empirical_context()
            probe.assert_not_called()
            saved = json.loads((folder / "context_gpu.json").read_text())
            backup = json.loads(
                (folder / "context_gpu.pre_empirical_reconciliation.json")
                .read_text()
            )
            self.assertEqual(backup, original)
            self.assertEqual(saved["confirmed_max_gpu_resident_n_ctx"], 2048)
            self.assertTrue(journal.cached("context_gpu")["success"])
            self.assertEqual(
                journal.state["phases"]["context_gpu"]["status"], "completed"
            )
            # Running twice cannot reclassify again or overwrite the backup.
            journal.reconcile_blocked_empirical_context()
            self.assertEqual(
                json.loads(
                    (folder / "context_gpu.pre_empirical_reconciliation.json")
                    .read_text()
                ), original,
            )


if __name__ == "__main__":
    unittest.main()
