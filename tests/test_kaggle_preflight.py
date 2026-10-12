"""Safe Kaggle transfer preview tests; no actual GGUF bytes fetched."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.kaggle_preflight import inspect_transfer_plan


class KaggleTransferPreflightTests(unittest.TestCase):
    def test_reported_sizes_not_confused_with_native_cuda_verification(self):
        gpu = {
            "ready": True,
            "memory": {"total_mib": 6141},
            "gpu": {"name": "Mock RTX 4050"},
        }
        candidates = [
            {
                "quantization": "fp16",
                "filename": "qwen-fp16-00001-of-00004.gguf",
                "shard_count": 4, "split_complete": True,
                "quality_priority": 0,
                "reported_download_bytes": 15_000_000_000,
                "reported_download_size_estimated": True,
            },
            {
                "quantization": "Q4_K_M",
                "filename": "qwen-Q4_K_M-00001-of-00002.gguf",
                "shard_count": 2, "split_complete": True,
                "quality_priority": 400,
                "reported_download_bytes": 4_000_000_000,
                "reported_download_size_estimated": False,
            },
            {
                "quantization": "Q6_K",
                "filename": "qwen-Q6_K.gguf",
                "shard_count": 1, "split_complete": True,
                "quality_priority": 300,
                "reported_download_bytes": None,
                "reported_download_size_estimated": None,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("calibration.kaggle_preflight.inventory_kaggle",
                      return_value=candidates) as listed,
                patch("kagglehub.model_download") as download,
                patch("calibration.gpu_context.probe_context") as probe,
            ):
                result = inspect_transfer_plan(
                    "qwen-2.5-7b-instruct",
                    disk_path=Path(directory), gpu_status=gpu,
                )
            listed.assert_called_once()
            download.assert_not_called()
            probe.assert_not_called()
            self.assertFalse(result["weights_downloaded"])
            self.assertFalse(result["native_gpu_probe_executed"])
            self.assertEqual(len(result["candidates"]), 3)
            self.assertTrue(
                result["candidates"][0]["reported_file_total_above_vram"]
            )
            self.assertEqual(
                result["candidates"][0]["cpu_gpu_load_feasibility"],
                "not_determined_from_file_sizes",
            )
            self.assertFalse(
                result["candidates"][1]["reported_file_total_above_vram"]
            )
            self.assertIsNone(
                result["candidates"][2]["reported_file_total_above_vram"]
            )

    def test_canonical_fp16_q8_excluded_by_architecture_without_download(self):
        gpu = {"ready": True, "memory": {"total_mib": 6141}}
        common = {
            "repo_id": "qwen-lm/qwen2.5",
            "kaggle_handle": "qwen-lm/qwen2.5/gguf/7b-instruct/1",
            "shard_count": 2,
            "split_complete": True,
        }
        candidates = [
            {**common, "quantization": "fp16",
             "filename": "Qwen2.5-7B-Instruct-fp16.gguf",
             "quality_priority": 900,
             "reported_download_bytes": 15_000_000_000},
            {**common, "quantization": "q8_0",
             "filename": "Qwen2.5-7B-Instruct-q8_0.gguf",
             "quality_priority": 800,
             "reported_download_bytes": 8_000_000_000},
            {**common, "quantization": "q6_k",
             "filename": "Qwen2.5-7B-Instruct-q6_k.gguf",
             "quality_priority": 700,
             "reported_download_bytes": 6_000_000_000},
        ]
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("calibration.kaggle_preflight.inventory_kaggle",
                      return_value=candidates),
                patch("kagglehub.model_download") as download,
            ):
                report = inspect_transfer_plan(
                    disk_path=Path(directory), gpu_status=gpu,
                )
            download.assert_not_called()
        rows = report["candidates"]
        self.assertTrue(rows[0]["pre_download_full_cuda_exclusion"])
        self.assertTrue(rows[1]["pre_download_full_cuda_exclusion"])
        self.assertFalse(rows[2]["pre_download_full_cuda_exclusion"])
        self.assertGreater(
            rows[0]["architecture_weight_floor"]["minimum_weight_mib"], 6141
        )
        self.assertGreater(
            rows[1]["architecture_weight_floor"]["minimum_weight_mib"], 6141
        )

    def test_incomplete_shards_are_recorded_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{
                "quantization": "Q5_K_M", "filename": "second-only.gguf",
                "shard_count": 2, "split_complete": False,
                "quality_priority": None,
                "reported_download_bytes": 200_000_000,
            }]
            with patch("calibration.kaggle_preflight.inventory_kaggle",
                       return_value=rows):
                result = inspect_transfer_plan(
                    gpu_status={"ready": True, "memory": {"total_mib": 6141}},
                    disk_path=Path(directory)
                )
            self.assertFalse(result["candidates"][0]["shards_complete"])
            self.assertFalse(result["weights_downloaded"])


if __name__ == "__main__":
    unittest.main()
