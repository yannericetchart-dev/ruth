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


def fake_mapping(selection):
    return {
        "schema_version": 1, "phase": "mapping",
        "model_id": selection["model_id"],
        "source": selection["source"],
        "filename": selection["filename"],
        "quantization": selection["quantization"],
        "model_path": str(Path(selection["local_path"]).resolve()),
        "status": "diagnostic_native_placement_validated",
        "selected_mapping": None, "profile_generated": False,
        "success": True, "trials": [
            {
                "mode": mode, "success": True,
                "native_placement": {"verified": True},
                "gpu_runtime_validation": {"verified": True},
                "measurements": {"vram_peak_sampled_mib": 5300},
            }
            for mode in ("block0_host", "attention_tensor_host")
        ],
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
            report["trials"][1]["gpu_runtime_validation"]["verified"] = False
            atomic_json(folder / "mapping.json", report)
            journal = Journal(folder, config)
            journal.phase_state("mapping", "completed", "mapping.json")
            self.assertIsNone(journal.cached("mapping"))
            report["trials"][1]["gpu_runtime_validation"]["verified"] = True
            report["trials"][1]["native_placement"]["verified"] = False
            atomic_json(folder / "mapping.json", report)
            self.assertIsNone(journal.cached("mapping"))

    def test_failed_mapping_blocks_then_retries_only_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, folder, selection = self.prepare(root)
            failure = fake_mapping(selection)
            failure.update({
                "success": False,
                "status": "blocked_unverified_native_mapping",
                "trials": [{"mode": "block0_host", "success": False}],
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
