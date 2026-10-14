"""Offline negative capacity bounds from public immutable Git CUDA records."""
import json
from io import StringIO
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.model_catalog import select_model
from ruth_runtime.offline_memory_fit import explain_offline_weights_fit
from ruth_runtime.launcher import main


class OfflineMemoryFitTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "catalog").mkdir(parents=True)
        for item in ("model_catalog.json", "nvidia_architecture_policy.json"):
            shutil.copyfile(PROJECT_ROOT / "catalog" / item,
                            self.root / "catalog" / item)
        choice = select_model("production")
        self.sha = choice.expected_sha256
        local = self.root / ".ruth-build/profiles/models"
        local.mkdir(parents=True)
        (local / (self.sha + ".json")).write_text(json.dumps({
            "schema": "ruth-model-runtime-memory-registry-v1",
            "model": {"gguf_sha256": self.sha},
            "runtime_measurements": {
                "cuda13.2": {
                    "architecture": "ada",
                    "driver_version": "595.99.02",
                    "cuda_backend": "13.2",
                    "llama_cpp_binding": "0.3.36",
                },
            },
        }))
        self.git = self.root / "profiles/measurements" / self.sha
        self.git.mkdir(parents=True)
        candidate = PROJECT_ROOT / "profiles" / "measurements" / self.sha
        fixtures = sorted(candidate.glob("*.json"))
        if not fixtures:
            self.skipTest("Committed Git memory evidence not checked out")
        shutil.copyfile(fixtures[0], self.git / fixtures[0].name)

    def report(self, **options):
        return explain_offline_weights_fit(project_root=self.root, **options)

    def test_six_gib_ada_is_rejected_from_git_without_probing_cuda(self):
        with patch("subprocess.run", side_effect=AssertionError("No hardware probe")), \
             patch("socket.create_connection",
                   side_effect=AssertionError("No network")):
            result = self.report(vram_gib=6)
        self.assertEqual(result["status"],
                         "full_gpu_weights_cannot_fit_nominal_bound")
        mem = result["memory"]
        self.assertEqual(mem["vram_nominal_total_mib"], 6144)
        self.assertEqual(mem["driver_reserved_mib"], 369.0)
        self.assertEqual(mem["cuda_init_observed_nvml_to_cuda_delta_mib"], 90.19)
        self.assertEqual(mem["weights_cuda_buffer_mib"], 5700.31)
        self.assertEqual(mem["vram_estimated_available_for_weights_mib"], 5684.81)
        self.assertEqual(mem["vram_weights_deficit_mib"], 15.50)
        self.assertFalse(result["compatible_for_full_gpu_weights"])
        self.assertFalse(result["no_gpu_queries"] is False)
        self.assertFalse(result["production_qualified"])
        self.assertTrue(result["evidence_files"])

    def test_cli_offline_fit_does_not_inspect_host_or_trigger_git(self):
        fake = {
            "status": "full_gpu_weights_cannot_fit_nominal_bound",
            "no_gpu_queries": True, "network_used": False,
            "compatible_for_full_gpu_weights": False,
        }
        with patch("sys.stdout", new_callable=StringIO) as output, \
             patch("ruth_runtime.offline_memory_fit.explain_offline_weights_fit",
                   return_value=fake) as calc, \
             patch("ruth_runtime.model_memory_profiles.build_profile_matrix",
                   side_effect=AssertionError("Should never inspect host")), \
             patch("ruth_runtime.profile_contribution.sync_profile_to_git",
                   side_effect=AssertionError("Should never publish")):
            status = main(["models", "profiles", "--offline-fit",
                           "--quantization", "Q4_K_L", "--vram-gib", "6"])
        self.assertEqual(status, 0)
        self.assertEqual(calc.call_args.kwargs["vram_gib"], 6)
        self.assertFalse(json.loads(output.getvalue())["compatible_for_full_gpu_weights"])

    def test_locally_cached_historical_host_values_are_not_live(self):
        local = self.root / ".ruth-build/profiles/models" / (self.sha + ".json")
        data = json.loads(local.read_text())
        data["host_snapshot"] = {
            "architecture": "ada", "driver_version": "595.99.02",
            "vram_class_gib": 6, "vram_observed_total_mib": 6141,
            "nvml_free_before_cuda_mib": 5773,
            "cuda_free_after_primary_ctx_init_mib": 5682.81,
            "historical_measurement": True,
        }
        local.write_text(json.dumps(data))
        result = self.report()
        self.assertEqual(result["status"], "full_gpu_weights_cannot_fit_nominal_bound")
        self.assertEqual(result["memory"]["historical_vram_observed_total_mib"], 6141)
        self.assertEqual(result["memory"]["historical_cuda_free_after_primary_ctx_init_mib"], 5682.81)
        self.assertIsNone(result["memory"]["measured_live_cuda_free_mib"])

    def test_eight_gib_is_possible_not_production_verified(self):
        outcome = self.report(vram_gib=8)
        self.assertEqual(outcome["status"], "weights_possible_context_unverified")
        self.assertIsNone(outcome["compatible_for_full_gpu_weights"])
        self.assertFalse(outcome["production_qualified"])

    def test_no_exact_driver_evidence_stays_unknown(self):
        outcome = self.report(vram_gib=6, driver_version="600.01.01")
        self.assertEqual(outcome["status"], "offline_fit_unverified")
        self.assertEqual(outcome["reason"],
                         "no_exact_architecture_cuda_driver_binding_evidence")
        self.assertIsNone(outcome["compatible_for_full_gpu_weights"])

    def test_missing_vram_size_stays_unknown(self):
        outcome = self.report()
        self.assertEqual(outcome["status"], "offline_fit_unverified")
        self.assertEqual(outcome["reason"], "host_vram_class_missing_or_forbidden")

    def test_unsupported_vram_class_rejected(self):
        outcome = self.report(vram_gib=4)
        self.assertEqual(outcome["reason"], "host_vram_class_missing_or_forbidden")

    def test_git_record_absent_never_fabricates_measurement(self):
        for path in self.git.iterdir():
            path.unlink()
        outcome = self.report(vram_gib=6)
        self.assertEqual(outcome["reason"], "no_exact_architecture_cuda_driver_binding_evidence")

    def test_invalid_git_blob_not_accepted(self):
        file = next(self.git.glob("*.json"))
        data = json.loads(file.read_text())
        data["production_qualified"] = True
        file.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.report(vram_gib=6)


if __name__ == "__main__":
    unittest.main()
