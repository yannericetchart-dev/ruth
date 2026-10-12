"""Offline checks: Kaggle shard totals versus VRAM, no download budgets."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from calibration.baseline_discovery import choose_baseline
from calibration.kaggle_transfer_guard import assess_kaggle_transfer


GPU = {
    "ready": True, "errors": [], "foreign_processes": [],
    "gpu": {"vendor": "nvidia", "name": "Mock CUDA",
            "uuid": "GPU-test", "index": 0},
    "memory": {"total_mib": 6141.0},
}
MODEL = "qwen-2.5-7b-instruct"
REPO = "qwen-lm/qwen2.5"
HANDLE = REPO + "/gguf/7b-instruct/1"


def candidate(quant: str, total: int | None, rank: int, shards: int = 1):
    if shards == 1:
        names = [f"Qwen2.5-7B-Instruct-{quant}.gguf"]
    else:
        names = [
            f"Qwen2.5-7B-Instruct-{quant}-{i:05d}-of-{shards:05d}.gguf"
            for i in range(1, shards + 1)
        ]
    sizes = None if total is None else [
        total // shards + (1 if i < total % shards else 0)
        for i in range(shards)
    ]
    return {
        "filename": names[0], "shard_filenames": names,
        "shard_source_file_paths": names,
        "shard_count": shards, "split_complete": True,
        "filename_matches_model_prefix": True,
        "quantization": quant, "quality_priority": rank,
        "repo_id": REPO, "kaggle_handle": HANDLE,
        "source_file_path": names[0],
        "reported_download_bytes": total,
        "reported_shard_sizes": (
            [{"bytes": n, "estimated": False} for n in sizes]
            if sizes is not None else [None] * shards
        ),
    }


class KaggleDiskGuardTests(unittest.TestCase):
    def test_checks_cache_and_destination_copies_without_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            model = candidate("Q5_K_M", 5_000_000_000, 1, 2)
            with patch("calibration.kaggle_transfer_guard.shutil.disk_usage",
                       return_value=SimpleNamespace(free=9_000_000_000)):
                failed = assess_kaggle_transfer(
                    model, Path(folder) / "models",
                    cache_root=Path(folder) / "cache",
                )
            self.assertFalse(failed["allowed"])
            self.assertEqual(
                failed["status"], "blocked_kaggle_insufficient_disk"
            )
            with patch("calibration.kaggle_transfer_guard.shutil.disk_usage",
                       return_value=SimpleNamespace(free=25_000_000_000)):
                passed = assess_kaggle_transfer(
                    model, Path(folder) / "models",
                    cache_root=Path(folder) / "cache",
                )
            self.assertTrue(passed["allowed"])
            self.assertEqual(passed["catalog_bytes"], 5_000_000_000)

    def test_existing_shards_only_count_missing_for_disk(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            model = candidate("Q5_K_M", 5_000_000_000, 1, 2)
            (path / model["shard_filenames"][0]).write_bytes(b"existing")
            with patch("calibration.kaggle_transfer_guard.shutil.disk_usage",
                       return_value=SimpleNamespace(free=15_000_000_000)):
                allowed = assess_kaggle_transfer(
                    model, path, cache_root=path / "cache"
                )
            self.assertTrue(allowed["allowed"])
            self.assertEqual(allowed["missing_shards"], 1)
            self.assertEqual(allowed["catalog_bytes"], 2_500_000_000)


class KaggleBaselineSizeTests(unittest.TestCase):
    def run_baseline(self, root, inventory, *, loader_bytes=5_000_000_000,
                     probe=None, source_spec=None):
        downloads = []
        attempts = []

        def fake_download(entry, folder):
            downloads.append(entry["quantization"])
            path = Path(folder) / entry["filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mock GGUF")
            return path

        def fake_probe(path, **kwargs):
            attempts.append(path.name)
            if probe is not None:
                return probe(path)
            return {
                "success": True, "architecture": "qwen2",
                "full_gpu_native_verification": {"verified": True},
                "measurements": {"minimal_inference_completed": True},
            }

        with (
            patch("calibration.baseline_discovery.GPUManager.calibration_status",
                  return_value=GPU),
            patch("calibration.kaggle_source.inventory_kaggle",
                  return_value=inventory),
            patch("calibration.kaggle_source.download_kaggle",
                  side_effect=fake_download),
            patch("calibration.baseline_discovery.inspect_gguf",
                  return_value={"weights_bytes": loader_bytes}),
            patch("calibration.baseline_discovery.probe_gpu_candidate",
                  side_effect=fake_probe),
            patch("calibration.kaggle_transfer_guard.shutil.disk_usage",
                  return_value=SimpleNamespace(free=100_000_000_000)),
        ):
            if source_spec is None:
                report = choose_baseline(
                    root=root, model_id=MODEL, source="kaggle",
                    kaggle_handle=HANDLE, repo_id=REPO,
                    persist_selection=False,
                )
            else:
                with patch(
                    "calibration.baseline_discovery.ModelManager.source_spec",
                    return_value=source_spec,
                ):
                    report = choose_baseline(
                        root=root, model_id=MODEL, source="kaggle",
                        kaggle_handle=HANDLE, repo_id=REPO,
                        persist_selection=False,
                    )
        return report, downloads, attempts

    def test_sums_four_and_three_shards_and_skips_over_vram(self):
        with tempfile.TemporaryDirectory() as folder:
            items = [
                candidate("fp16", 15_000_000_000, 0, 4),
                candidate("q8_0", 8_000_000_000, 1, 3),
                candidate("q6_k", 6_000_000_000, 2, 2),
            ]
            report, downloads, attempts = self.run_baseline(
                Path(folder), items, loader_bytes=5_000_000_000
            )
        self.assertTrue(report["success"])
        self.assertEqual(downloads, ["q6_k"])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(
            [r["status"] for r in report["attempts"]],
            ["excluded_catalog_gguf_size_exceeds_total_vram",
             "excluded_catalog_gguf_size_exceeds_total_vram",
             "selected_full_gpu_baseline"],
        )
        self.assertEqual(report["attempts"][0]["catalog_gguf_total_mib"],
                         round(15_000_000_000 / 1048576, 2))

    def test_q6_oom_then_q5_downloads_in_same_run_without_budget(self):
        def probe(path):
            if "q6_k" in path.name:
                return {
                    "success": False,
                    "failure_hint": "explicit_cuda_oom_evidence",
                    "suspected_cuda_oom": True, "error": "CUDA OOM",
                }
            return {
                "success": True, "architecture": "qwen2",
                "full_gpu_native_verification": {"verified": True},
                "measurements": {"minimal_inference_completed": True},
            }

        with tempfile.TemporaryDirectory() as folder:
            items = [
                candidate("q6_k", 6_000_000_000, 0, 2),
                candidate("q5_k_m", 5_000_000_000, 1, 2),
            ]
            report, downloads, attempts = self.run_baseline(
                Path(folder), items, probe=probe
            )
        self.assertTrue(report["success"])
        self.assertEqual(downloads, ["q6_k", "q5_k_m"])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(report["baseline"]["quantization"], "q5_k_m")

    def test_unknown_size_blocks_without_downloading_or_skipping(self):
        with tempfile.TemporaryDirectory() as folder:
            items = [candidate("fp16", None, 0),
                     candidate("q5_k_m", 5_000_000_000, 1)]
            report, downloads, attempts = self.run_baseline(
                Path(folder), items
            )
        self.assertFalse(report["success"])
        self.assertEqual(downloads, [])
        self.assertEqual(attempts, [])
        self.assertEqual(
            report["attempts"][0]["status"], "unverified_kaggle_file_sizes"
        )

    def test_filter_is_not_qwen_specific(self):
        with tempfile.TemporaryDirectory() as folder:
            items = [candidate("fp16", 15_000_000_000, 0, 4),
                     candidate("q5_k_m", 5_000_000_000, 1)]
            # The model's declared architecture is irrelevant to size gate.
            custom = {
                "source": "kaggle", "repository": REPO,
                "kaggle_handle": HANDLE,
                "gguf_prefix": "Another-",
                "architecture": "qwen2",
            }
            report, downloads, attempts = self.run_baseline(
                Path(folder), items, source_spec=custom
            )
        self.assertTrue(report["success"])
        self.assertEqual(downloads, ["q5_k_m"])
        self.assertEqual(
            report["attempts"][0]["status"],
            "excluded_catalog_gguf_size_exceeds_total_vram",
        )


if __name__ == "__main__":
    unittest.main()
