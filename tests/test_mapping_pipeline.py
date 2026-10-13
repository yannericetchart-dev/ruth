"""Offline, resumable phase-2 pipeline tests; no GPU worker or downloads."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.profile_pipeline import Journal, atomic_json, run_pipeline


def identity():
    return {
        "profile_key": "phase2-mapping-test",
        "choices": {
            "model_id": "qwen-2.5-7b-instruct",
            "source": "bartowski",
            "repository": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "engine": "llama.cpp", "method": "llama_cpp_python",
            "kaggle_handle": None,
        },
        "hardware": {
            "backend": "cuda", "gpu_name": "Mock RTX 4050",
            "vram_total_mib": 6141, "ram_total_mib": 15495,
        },
        "reference_runtime": {
            "baseline_n_ctx": 64, "n_batch": 32,
            "context_confirm_rounds": 2,
        },
    }


def evidence(group_id):
    return {
        "mode": group_id, "success": True,
        "native_placement": {"verified": True},
        "gpu_runtime_validation": {
            "verified": True,
            "kv": {
                "verified_host_resident": group_id == "baseline_host_kv",
                "verified_gpu_resident": group_id != "baseline_host_kv",
            },
        },
        "kv_target": "host" if group_id == "baseline_host_kv" else "gpu",
        "offload_kqv": group_id != "baseline_host_kv",
        "measurements": {
            "vram_peak_sampled_mib": 5300,
            "host_peak_rss_mib": 1200,
            "prefill_tokens_per_s": 80,
            "decode_tokens_per_s": 35,
        },
    }


def fake_mapping(selection):
    ids = ["group:attn_k.weight", "group:ffn_down.weight"]
    return {
        "schema_version": 4, "phase": "mapping",
        "model_id": selection["model_id"],
        "source": selection["source"],
        "filename": selection["filename"],
        "quantization": selection["quantization"],
        "model_path": str(Path(selection["local_path"]).resolve()),
        "reference_full_cuda_n_ctx": 2048,
        "probe_n_ctx": 4096, "probe_tokens": 128,
        "baseline_kv_target": "host", "candidate_kv_target": "gpu",
        "baseline_offload_kqv": False, "candidate_offload_kqv": True,
        "repeat_processes_per_variant": 3,
        "tested_group_ids": ids,
        "groups": [
            {"id": name, "eligible_for_trial": True} for name in ids
        ],
        "baseline_trials": [
            evidence("baseline_host_kv") for _ in range(3)
        ],
        "baseline_statistics": {"prefill_tokens_per_s": {"median": 80}},
        "group_evaluations": [
            {
                "group_id": name, "status": "rejected_performance_regression",
                "comparison": {"verdict": "rejected_performance_regression"},
                "trials": [evidence(name) for _ in range(3)],
            }
            for name in ids
        ],
        "status": "diagnostic_group_comparison_completed",
        "selected_mapping": None, "profile_generated": False,
        "success": True,
    }

class MappingPipelineTests(unittest.TestCase):
    def prepare(self, root):
        config = identity()
        folder = root / config["profile_key"]
        folder.mkdir()
        model = root / "baseline.gguf"
        model.write_bytes(b"test")
        selection = {
            "model_id": config["choices"]["model_id"],
            "source": config["choices"]["source"],
            "repository": config["choices"]["repository"],
            "local_path": str(model), "filename": model.name,
            "quantization": "Q5_K_L",
            "native_verified_all_gpu": True,
        }
        reports = {
            "inventory": {
                "inventory_only": True,
                "inventory": [{"filename": model.name}],
                "unranked_quantizations": [],
            },
            "baseline": {"success": True, "baseline": selection},
            "layer_matrix": {
                "success": True, "status": "enumeration_only",
                "schema_version": 2, "filename": model.name,
                "model_path": str(model.resolve()),
                "block_count": 2, "layers": [{"index": 0}, {"index": 1}],
                "serialized_weight_bytes": 12345,
                "cpu_offload_executed": False,
            },
            "context_gpu": {
                "success": True, "confirmed_max_gpu_resident_n_ctx": 2048,
            },
        }
        journal = Journal(folder, config)
        for name, report in reports.items():
            atomic_json(folder / f"{name}.json", report)
            journal.phase_state(name, "completed", f"{name}.json")
        return config, folder, selection

    def test_mapping_runs_once_and_resumes_without_native_reprobe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, folder, baseline = self.prepare(root)
            with patch(
                "calibration.mapping_phase.run_mapping_phase",
                return_value=fake_mapping(baseline),
            ) as worker:
                first = run_pipeline(
                    config, until="mapping", output_root=root
                )
                second = run_pipeline(
                    config, until="mapping", output_root=root
                )
            self.assertTrue(first["success"])
            self.assertTrue(second["success"])
            self.assertEqual(worker.call_count, 1)
            self.assertIsNotNone(Journal(folder, config).cached("mapping"))
            self.assertEqual(
                json.loads((folder / "state.json").read_text())["phases"][
                    "mapping"
                ]["status"], "completed",
            )
            self.assertIsNone(
                json.loads((folder / "mapping.json").read_text())[
                    "selected_mapping"
                ]
            )

    def test_unverified_mapping_never_cached_as_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, folder, baseline = self.prepare(root)
            report = fake_mapping(baseline)
            report["group_evaluations"][1]["trials"][2]["gpu_runtime_validation"]["verified"] = False
            atomic_json(folder / "mapping.json", report)
            journal = Journal(folder, config)
            journal.phase_state("mapping", "completed", "mapping.json")
            self.assertIsNone(journal.cached("mapping"))
            report["group_evaluations"][1]["trials"][2]["gpu_runtime_validation"]["verified"] = True
            report["group_evaluations"][1]["trials"][1]["native_placement"]["verified"] = False
            atomic_json(folder / "mapping.json", report)
            self.assertIsNone(journal.cached("mapping"))

    def test_failed_single_group_counts_as_completed_survey_not_successful_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, folder, selection = self.prepare(root)
            report = fake_mapping(selection)
            group_id = report["tested_group_ids"][1]
            report["group_evaluations"][1] = {
                "group_id": group_id,
                "trials": [{
                    "mode": group_id, "success": False,
                    "failure_class": "worker_protocol_error",
                }],
                "comparison": None,
                "statistics": None,
                "status": "native_probe_failed",
                "failure_class": "worker_protocol_error",
            }
            report["native_failed_group_count"] = 1
            atomic_json(folder / "mapping.json", report)
            journal = Journal(folder, config)
            journal.phase_state("mapping", "completed", "mapping.json")
            validated = journal.cached("mapping")
            self.assertIsNotNone(validated)
            self.assertTrue(validated["success"])
            self.assertEqual(
                validated["group_evaluations"][1]["status"],
                "native_probe_failed",
            )
            self.assertIsNone(validated["selected_mapping"])

    def test_failed_mapping_blocks_then_retries_only_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, folder, selection = self.prepare(root)
            failure = fake_mapping(selection)
            failure.update({
                "success": False,
                "status": "blocked_baseline_native_benchmark",
                "group_evaluations": [{"group_id": "group:attn_k.weight",
                                     "trials": [{"success": False}]}],
                "error": "Native placement unverified",
            })
            with patch(
                "calibration.mapping_phase.run_mapping_phase",
                side_effect=[failure, fake_mapping(selection)],
            ) as worker:
                first = run_pipeline(config, until="mapping", output_root=root)
                second = run_pipeline(config, until="mapping", output_root=root)
            self.assertFalse(first["success"])
            self.assertEqual(first["blocked_phase"], "mapping")
            self.assertTrue(second["success"])
            self.assertEqual(worker.call_count, 2)
            state = json.loads((folder / "state.json").read_text())
            self.assertEqual(
                [state["phases"][n]["status"] for n in
                 ("inventory", "baseline", "layer_matrix", "context_gpu", "mapping")],
                ["completed"] * 5,
            )


if __name__ == "__main__":
    unittest.main()
