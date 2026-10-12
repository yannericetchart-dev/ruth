"""Offline, fail-closed tests for consent-based Kaggle GGUF downloads."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from calibration.kaggle_transfer_guard import assess_kaggle_transfer
from calibration.baseline_discovery import choose_baseline


MODEL = "qwen-2.5-7b-instruct"
REPO = "qwen-lm/qwen2.5"
HANDLE = REPO + "/gguf/7b-instruct/1"
GPU = {
    "ready": True, "errors": [], "foreign_processes": [],
    "gpu": {"vendor": "nvidia", "name": "Mock RTX 4050",
            "uuid": "GPU-test", "index": 0},
    "memory": {"total_mib": 6141.0},
}


def candidate(quant, size, rank):
    name = f"Qwen2.5-7B-Instruct-{quant}.gguf"
    return {
        "filename": name, "quantization": quant,
        "quality_priority": rank,
        "kaggle_handle": HANDLE, "repo_id": REPO,
        "source_file_path": name,
        "shard_filenames": [name],
        "shard_source_file_paths": [name],
        "shard_count": 1,
        "split_complete": True,
        "reported_shard_sizes": (
            [{"bytes": size, "estimated": False}] if size is not None else [None]
        ),
    }


class KaggleTransferGuardTests(unittest.TestCase):
    def test_no_budget_refuses_transfer_before_asking_cache_or_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("calibration.kaggle_transfer_guard.shutil.disk_usage") as disk:
                outcome = assess_kaggle_transfer(
                    candidate("fp16", 15_000_000_000, 0),
                    Path(temp), remaining_budget_bytes=None,
                )
            disk.assert_not_called()
        self.assertFalse(outcome["allowed"])
        self.assertEqual(
            outcome["status"], "blocked_kaggle_download_approval_required"
        )

    def test_budget_and_unknown_size_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            insufficient = assess_kaggle_transfer(
                candidate("fp16", 15_000_000_000, 0),
                root, remaining_budget_bytes=10_000_000_000,
            )
            self.assertEqual(
                insufficient["status"], "blocked_kaggle_download_budget"
            )
            missing_size = assess_kaggle_transfer(
                candidate("Q8_0", None, 100),
                root, remaining_budget_bytes=10_000_000_000,
            )
            self.assertEqual(
                missing_size["status"], "blocked_kaggle_unknown_transfer_size"
            )

    def test_cache_and_copy_on_same_disk_are_both_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = candidate("Q5_K_M", 5_000_000_000, 400)
            with patch(
                "calibration.kaggle_transfer_guard.shutil.disk_usage",
                return_value=SimpleNamespace(free=10_000_000_000),
            ):
                blocked = assess_kaggle_transfer(
                    item, root / "model",
                    remaining_budget_bytes=6_000_000_000,
                    min_free_mib=0, cache_root=root / "cache",
                )
            self.assertFalse(blocked["allowed"])
            self.assertEqual(
                blocked["status"], "blocked_kaggle_insufficient_disk"
            )
            with patch(
                "calibration.kaggle_transfer_guard.shutil.disk_usage",
                return_value=SimpleNamespace(free=25_000_000_000),
            ):
                allowed = assess_kaggle_transfer(
                    item, root / "model",
                    remaining_budget_bytes=6_000_000_000,
                    min_free_mib=1024, cache_root=root / "cache",
                )
            self.assertTrue(allowed["allowed"])
            self.assertEqual(allowed["catalog_transfer_bytes"], 5_000_000_000)
            self.assertTrue(allowed["same_disk_for_cache_and_copy"])

    def test_existing_shards_reduce_transfer_estimate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            names = ["qwen-q5_k_m-00001-of-00002.gguf",
                     "qwen-q5_k_m-00002-of-00002.gguf"]
            (root / names[0]).write_bytes(b"already downloaded")
            item = candidate("Q5_K_M", 1_000_000_000, 400)
            item["filename"] = names[0]
            item["shard_filenames"] = names
            item["reported_shard_sizes"] = [
                {"bytes": 1_000_000_000}, {"bytes": 2_000_000_000}
            ]
            with patch(
                "calibration.kaggle_transfer_guard.shutil.disk_usage",
                return_value=SimpleNamespace(free=25_000_000_000),
            ):
                approved = assess_kaggle_transfer(
                    item, root, remaining_budget_bytes=2_500_000_000,
                    min_free_mib=0, cache_root=root / "cache",
                )
            self.assertTrue(approved["allowed"])
            self.assertEqual(approved["missing_shards"], 1)
            self.assertEqual(approved["catalog_transfer_bytes"], 2_000_000_000)
            self.assertEqual(
                approved["remaining_budget_after_bytes"], 500_000_000
            )


class KaggleBudgetPipelineTests(unittest.TestCase):
    def run_baseline(self, root, inventory, *, approval=None, reserve=0,
                     weights_bytes=7_000_000_000):
        with (
            patch("calibration.baseline_discovery.GPUManager.calibration_status",
                  return_value=GPU),
            patch("calibration.kaggle_source.inventory_kaggle",
                  return_value=inventory),
            patch("calibration.kaggle_source.download_kaggle") as downloader,
            patch("calibration.baseline_discovery.inspect_gguf",
                  return_value={"weights_bytes": weights_bytes}),
            patch("calibration.baseline_discovery.probe_gpu_candidate") as gpu,
        ):
            def download(item, dest):
                path = Path(dest) / item["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"simulated GGUF")
                return path
            downloader.side_effect = download
            result = choose_baseline(
                root=root, model_id=MODEL, source="kaggle",
                kaggle_handle=HANDLE, repo_id=REPO,
                persist_selection=False,
                kaggle_max_download_mib=approval,
                kaggle_min_free_mib=reserve,
            )
            return result, downloader.call_count, gpu.call_count

    def test_no_approval_stops_at_fp16_without_downloading_or_skipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [
                candidate("fp16", 15_000_000_000, 0),
                candidate("Q5_K_M", 5_000_000_000, 300),
            ]
            result, downloaded, probed = self.run_baseline(
                Path(tmp), inventory,
            )
            self.assertFalse(result["success"])
            self.assertEqual(downloaded, 0)
            self.assertEqual(probed, 0)
            self.assertEqual(len(result["attempts"]), 1)
            self.assertEqual(result["blocked_candidate"]["quantization"], "fp16")
            self.assertEqual(
                result["attempts"][0]["status"],
                "blocked_kaggle_download_approval_required",
            )

    def test_cumulative_budget_stops_at_next_high_quality_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [
                candidate("fp16", 1_000_000, 0),
                candidate("Q8_0", 1_000_000, 100),
                candidate("Q5_K_M", 500_000, 300),
            ]
            result, downloaded, probed = self.run_baseline(
                Path(tmp), inventory, approval=1,
            )
            self.assertFalse(result["success"])
            self.assertEqual(downloaded, 1)
            self.assertEqual(probed, 0)
            self.assertEqual(
                [a["status"] for a in result["attempts"]],
                ["excluded_weight_payload_exceeds_total_vram",
                 "blocked_kaggle_download_budget"],
            )
            self.assertEqual(result["blocked_candidate"]["quantization"], "Q8_0")

    def test_unknown_size_blocks_even_with_large_explicit_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, downloaded, probed = self.run_baseline(
                Path(tmp), [candidate("fp16", None, 0)],
                approval=16_000,
            )
            self.assertFalse(result["success"])
            self.assertEqual(downloaded, 0)
            self.assertEqual(probed, 0)
            self.assertEqual(
                result["attempts"][0]["status"],
                "blocked_kaggle_unknown_transfer_size",
            )


if __name__ == "__main__":
    unittest.main()
