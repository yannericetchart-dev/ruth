"""Metadata-only Kaggle GGUF preflight: sum shards; no model download."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibration.kaggle_preflight import inspect_transfer_plan


class KaggleTransferPreflightTests(unittest.TestCase):
    def test_sums_reported_shard_sizes_and_compares_to_vram(self):
        rows = [
            {
                "quantization": "fp16",
                "filename": "qwen-fp16-00001-of-00004.gguf",
                "shard_count": 4, "split_complete": True,
                "quality_priority": 0,
                "reported_download_bytes": 15_000_000_000,
                "reported_download_size_estimated": True,
            },
            {
                "quantization": "q6_k",
                "filename": "qwen-q6-00001-of-00002.gguf",
                "shard_count": 2, "split_complete": True,
                "quality_priority": 1,
                "reported_download_bytes": 6_000_000_000,
                "reported_download_size_estimated": False,
            },
            {
                "quantization": "q5_k_m",
                "filename": "qwen-q5-00001-of-00002.gguf",
                "shard_count": 2, "split_complete": True,
                "quality_priority": 2,
                "reported_download_bytes": None,
                "reported_download_size_estimated": None,
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch("calibration.kaggle_preflight.inventory_kaggle",
                      return_value=rows),
                patch("kagglehub.model_download") as download,
            ):
                report = inspect_transfer_plan(
                    gpu_status={"ready": True, "memory": {
                        "total_mib": 6141
                    }},
                    disk_path=Path(temp),
                )
            download.assert_not_called()
        entries = report["candidates"]
        self.assertTrue(entries[0]["reported_file_total_above_vram"])
        self.assertFalse(entries[1]["reported_file_total_above_vram"])
        self.assertIsNone(entries[2]["reported_file_total_above_vram"])
        self.assertEqual(entries[0]["reported_download_bytes"], 15_000_000_000)
        self.assertFalse(report["weights_downloaded"])
        self.assertFalse(report["native_gpu_probe_executed"])

    def test_incomplete_shard_group_is_never_promoted(self):
        rows = [{
            "quantization": "q5_k_m",
            "filename": "qwen-q5-00002-of-00002.gguf",
            "shard_count": 2, "split_complete": False,
            "quality_priority": None,
            "reported_download_bytes": 200_000_000,
        }]
        with tempfile.TemporaryDirectory() as temp:
            with patch("calibration.kaggle_preflight.inventory_kaggle",
                       return_value=rows):
                report = inspect_transfer_plan(
                    gpu_status={"ready": True, "memory": {
                        "total_mib": 6141
                    }},
                    disk_path=Path(temp),
                )
        self.assertFalse(report["candidates"][0]["shards_complete"])
        self.assertFalse(report["weights_downloaded"])


if __name__ == "__main__":
    unittest.main()
