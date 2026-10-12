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
ARCH_REF = {
    "repo": "Qwen/Qwen2.5-7B-Instruct",
    "config": {
        "model_type": "qwen2", "num_hidden_layers": 28,
        "hidden_size": 3584, "intermediate_size": 18944,
        "num_attention_heads": 28, "num_key_value_heads": 4,
        "vocab_size": 152064,
    },
}

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
                    Path(temp), candidate_budget_bytes=None,
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
                root, candidate_budget_bytes=10_000_000_000,
            )
            self.assertEqual(
                insufficient["status"], "blocked_kaggle_download_budget"
            )
            missing_size = assess_kaggle_transfer(
                candidate("Q8_0", None, 100),
                root, candidate_budget_bytes=10_000_000_000,
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
                    candidate_budget_bytes=6_000_000_000,
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
                    candidate_budget_bytes=6_000_000_000,
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
                    item, root, candidate_budget_bytes=2_500_000_000,
                    min_free_mib=0, cache_root=root / "cache",
                )
            self.assertTrue(approved["allowed"])
            self.assertEqual(approved["missing_shards"], 1)
            self.assertEqual(approved["catalog_transfer_bytes"], 2_000_000_000)
            self.assertEqual(
                approved["candidate_budget_unused_bytes"], 500_000_000
            )


class KaggleBudgetPipelineTests(unittest.TestCase):
    def run_baseline(self, root, inventory, *, approval=None, reserve=0,
                     weights_bytes=7_000_000_000):
        with (
            patch("calibration.baseline_discovery.GPUManager.calibration_status",
                  return_value=GPU),
            patch("calibration.kaggle_source.inventory_kaggle",
                  return_value=inventory),
            patch("calibration.kaggle_architecture_bound.read_architecture_reference",
                  return_value=ARCH_REF),
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

    def test_no_approval_skips_impossible_fp16_q8_then_blocks_q6(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [
                candidate("fp16", 15_000_000_000, 0),
                candidate("q8_0", 8_000_000_000, 100),
                candidate("q6_k", 6_000_000_000, 200),
                candidate("q5_k_m", 5_000_000_000, 300),
            ]
            result, downloaded, probed = self.run_baseline(
                Path(tmp), inventory,
            )
            self.assertFalse(result["success"])
            self.assertEqual(downloaded, 0)
            self.assertEqual(probed, 0)
            self.assertEqual(len(result["attempts"]), 3)
            self.assertEqual(
                [a["status"] for a in result["attempts"]],
                ["excluded_architecture_weight_floor_exceeds_vram",
                 "excluded_architecture_weight_floor_exceeds_vram",
                 "blocked_kaggle_download_approval_required"],
            )
            self.assertEqual(result["blocked_candidate"]["quantization"], "q6_k")
            self.assertGreater(
                result["attempts"][0]["architecture_weight_floor"][
                    "minimum_weight_mib"], 6141
            )
            self.assertGreater(
                result["attempts"][1]["architecture_weight_floor"][
                    "minimum_weight_mib"], 6141
            )
            self.assertEqual(
                result["attempts"][2]["status"],
                "blocked_kaggle_download_approval_required",
            )

    def test_explicit_large_budget_never_downloads_four_fp16_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp16 = candidate("fp16", 15_000_000_000, 0)
            names = [
                f"Qwen2.5-7B-Instruct-fp16-{i:05d}-of-00004.gguf"
                for i in range(1, 5)
            ]
            fp16.update({
                "filename": names[0],
                "shard_filenames": names,
                "shard_source_file_paths": names,
                "shard_count": 4,
                "reported_download_bytes": 15_000_000_000,
                "reported_shard_sizes": [
                    {"bytes": 3_750_000_000, "estimated": False}
                    for _ in names
                ],
            })
            q8 = candidate("q8_0", 8_000_000_000, 100)
            q6 = candidate("q6_k", 6_000_000_000, 200)
            result, downloaded, probed = self.run_baseline(
                Path(tmp), [fp16, q8, q6], approval=16000,
            )
            self.assertFalse(result["success"])
            self.assertEqual(downloaded, 1)
            self.assertEqual(probed, 0)
            self.assertEqual(
                [entry["status"] for entry in result["attempts"]],
                ["excluded_architecture_weight_floor_exceeds_vram",
                 "excluded_architecture_weight_floor_exceeds_vram",
                 "excluded_weight_payload_exceeds_total_vram"],
            )
            self.assertIsNone(result["baseline"])

    def test_budget_resets_per_quantization_not_cumulatively(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [
                candidate("Q6_K", 1_000_000, 200),
                candidate("Q5_K_M", 1_000_000, 300),
                candidate("Q4_K_M", 2_000_000, 400),
            ]
            result, downloaded, probed = self.run_baseline(
                Path(tmp), inventory, approval=1,
            )
            self.assertFalse(result["success"])
            # Two independently affordable candidates, then a single
            # candidate larger than the 1 MiB per-candidate limit.
            self.assertEqual(downloaded, 2)
            self.assertEqual(probed, 0)
            self.assertEqual(
                [a["status"] for a in result["attempts"]],
                ["excluded_weight_payload_exceeds_total_vram",
                 "excluded_weight_payload_exceeds_total_vram",
                 "blocked_kaggle_download_budget"],
            )
            self.assertEqual(result["blocked_candidate"]["quantization"], "Q4_K_M")

    def test_q6_oom_then_q5_full_cuda_selected_with_same_per_candidate_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = [
                candidate("Q6_K", 1_000_000, 200),
                candidate("Q5_K_M", 1_000_000, 300),
            ]
            downloaded = []
            tested = []

            def download(item, dest):
                downloaded.append(item["quantization"])
                path = Path(dest) / item["filename"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"mock GGUF")
                return path

            def probe(path, **kwargs):
                tested.append(path.name)
                if "Q6_K" in path.name:
                    return {
                        "success": False,
                        "failure_hint": "explicit_cuda_oom_evidence",
                        "suspected_cuda_oom": True,
                        "error": "native CUDA OOM",
                    }
                return {
                    "success": True, "architecture": "qwen2",
                    "full_gpu_native_verification": {"verified": True},
                    "measurements": {"minimal_inference_completed": True},
                }

            with (
                patch("calibration.baseline_discovery.GPUManager.calibration_status",
                      return_value=GPU),
                patch("calibration.kaggle_source.inventory_kaggle",
                      return_value=models),
                patch("calibration.kaggle_architecture_bound.read_architecture_reference",
                      return_value=ARCH_REF),
                patch("calibration.kaggle_source.download_kaggle",
                      side_effect=download),
                patch("calibration.baseline_discovery.inspect_gguf",
                      return_value={"weights_bytes": 5_000_000_000}),
                patch("calibration.baseline_discovery.probe_gpu_candidate",
                      side_effect=probe),
            ):
                report = choose_baseline(
                    root=root, model_id=MODEL, source="kaggle",
                    kaggle_handle=HANDLE, repo_id=REPO,
                    persist_selection=False,
                    kaggle_max_download_mib=1,
                    kaggle_min_free_mib=0,
                )
            self.assertTrue(report["success"])
            self.assertEqual(
                [a["status"] for a in report["attempts"]],
                ["failed_full_gpu_native_probe", "selected_full_gpu_baseline"],
            )
            self.assertEqual(downloaded, ["Q6_K", "Q5_K_M"])
            self.assertEqual(len(tested), 2)
            self.assertEqual(report["baseline"]["quantization"], "Q5_K_M")
            self.assertEqual(report["kaggle_budget_scope"], "per_candidate")

    def test_unknown_size_blocks_even_with_large_explicit_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, downloaded, probed = self.run_baseline(
                Path(tmp), [candidate("Q6_K", None, 200)],
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
