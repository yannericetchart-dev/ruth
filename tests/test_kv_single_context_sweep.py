"""No-CUDA regression tests for Ruth single-tensor +256 context sweep."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.kv_single_context_sweep import run_single_context_sweep


def make_census(root):
    model = root / "fake.gguf"
    model.write_bytes(b"fake")
    census = root / "census.json"
    census.write_text(json.dumps({
        "schema": "ruth-phase2-single-tensor-census-v1",
        "completed": True,
        "parameters": {
            "n_ctx": 256, "model_id": "gemma-2-9b-it",
            "model_path": str(model),
        },
    }))
    return census


class ContextSweepTests(unittest.TestCase):
    def test_moves_by_256_and_stops_first_zero_inference_context(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            census = make_census(root)
            called = []
            def runner(*, target_ctx, **kwargs):
                called.append(target_ctx)
                value = 5 if target_ctx == 512 else 0
                return {
                    "completed": True,
                    "summary": {
                        "ready_for_phase3": True,
                        "load_success": value,
                        "inference_success": value,
                        "remaining_loads": 0,
                        "remaining_inference": 0,
                    },
                }
            with patch("calibration.kv_single_context_sweep.inspect_gguf",
                       return_value={
                           "metadata": {
                               "general.architecture": "gemma2",
                               "gemma2.context_length": 2048,
                           },
                       }):
                result = run_single_context_sweep(
                    census_path=census, output_dir=root / "stage2",
                    stage_runner=runner,
                )
            self.assertEqual(called, [512, 768])
            self.assertEqual(result["status"], "stopped_no_valid_singletons")
            self.assertTrue(result["completed"])
            self.assertTrue(result["ready_for_phase3"])
            self.assertEqual(result["last_valid_target_ctx"], 512)
            self.assertEqual(
                result["first_target_without_valid_inference"], 768
            )
            self.assertEqual(len(result["contexts"]), 2)

    def test_incomplete_context_does_not_trigger_stage_three(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            census = make_census(root)
            called = []
            def runner(*, target_ctx, **kwargs):
                called.append(target_ctx)
                return {
                    "completed": False,
                    "summary": {
                        "ready_for_phase3": False,
                        "load_success": 0,
                        "inference_success": 0,
                        "remaining_loads": 6,
                        "remaining_inference": 0,
                    },
                }
            with patch("calibration.kv_single_context_sweep.inspect_gguf",
                       return_value={
                           "metadata": {
                               "general.architecture": "gemma2",
                               "gemma2.context_length": 2048,
                           },
                       }):
                result = run_single_context_sweep(
                    census_path=census, output_dir=root / "stage2",
                    stage_runner=runner,
                )
            self.assertEqual(called, [512])
            self.assertFalse(result["completed"])
            self.assertFalse(result["ready_for_phase3"])
            self.assertEqual(result["status"], "partial_context_checkpoint")
            self.assertIsNone(result["first_target_without_valid_inference"])
            self.assertEqual(result["next_ctx"], 512)

    def test_context_limit_is_respected_without_extra_overshoot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            census = make_census(root)
            called = []
            def runner(*, target_ctx, **kwargs):
                called.append(target_ctx)
                return {
                    "completed": True,
                    "summary": {
                        "ready_for_phase3": True,
                        "load_success": 1,
                        "inference_success": 1,
                        "remaining_loads": 0,
                        "remaining_inference": 0,
                    },
                }
            with patch("calibration.kv_single_context_sweep.inspect_gguf",
                       return_value={
                           "metadata": {
                               "general.architecture": "gemma2",
                               "gemma2.context_length": 768,
                           },
                       }):
                result = run_single_context_sweep(
                    census_path=census, output_dir=root / "stage2",
                    stage_runner=runner,
                )
            self.assertEqual(called, [512, 768])
            self.assertEqual(
                result["status"], "architecture_context_limit_reached"
            )
            self.assertIsNone(result["first_target_without_valid_inference"])
            self.assertEqual(result["last_valid_target_ctx"], 768)

    def test_resume_keeps_same_source_and_rejects_mutated_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            census = make_census(root)
            output = root / "stage2"
            def runner(**kwargs):
                return {
                    "completed": False,
                    "summary": {
                        "ready_for_phase3": False,
                        "load_success": 1,
                        "inference_success": 0,
                        "remaining_loads": 5,
                        "remaining_inference": 1,
                    },
                }
            with patch("calibration.kv_single_context_sweep.inspect_gguf",
                       return_value={
                           "metadata": {
                               "general.architecture": "gemma2",
                               "gemma2.context_length": 1024,
                           },
                       }):
                run_single_context_sweep(
                    census_path=census, output_dir=output,
                    stage_runner=runner,
                )
                original = json.loads(census.read_text())
                original["new_parameter"] = "changed"
                census.write_text(json.dumps(original))
                with self.assertRaisesRegex(
                    ValueError, "Checkpoint sweep incompatibile"
                ):
                    run_single_context_sweep(
                        census_path=census, output_dir=output,
                        stage_runner=runner,
                    )


if __name__ == "__main__":
    unittest.main()
