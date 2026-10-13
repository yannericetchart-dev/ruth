"""Offline tests: homologous GGUF groups, native CUDA proof, baseline comparison."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from calibration.mapping_grouping import (
    group_inventory, choose_group_trials, native_group_trial, MIB,
)
from calibration.mapping_phase import (
    benchmark_context, probe_native_mapping, run_mapping_phase,
    _median_stats, _compare_with_baseline,
)
from calibration.mapping_native_validation import verify_override_logs


def layout(arch="qwen2", blocks=2):
    entries = [
        {"name": "token_embd.weight", "bytes": 300 * MIB,
         "shape": [1024, 1024], "type": "Q5_K", "block": None},
    ]
    for block in range(blocks):
        for suffix, mib in (
            ("attn_k.weight", 10), ("attn_q.weight", 15),
            ("ffn_down.weight", 24), ("attn_norm.weight", 1),
        ):
            entries.append({
                "name": f"blk.{block}.{suffix}",
                "bytes": mib * MIB,
                "shape": [1024, 1024] if "norm" not in suffix else [1024],
                "type": "Q5_K", "block": block,
            })
    if arch != "gemma2":
        entries.append({
            "name": "output.weight", "bytes": 300 * MIB,
            "shape": [1024, 1024], "type": "Q5_K", "block": None,
        })
        entries.append({
            "name": "rope_freqs.weight", "bytes": 512,
            "shape": [128], "type": "F32", "block": None,
        })
    return {
        "metadata": {
            "general.architecture": arch,
            f"{arch}.block_count": blocks,
        },
        "tensors": entries,
        "weights_bytes": sum(x["bytes"] for x in entries),
    }


def native_log(trial, broken=None):
    rows = ["llama_model_load_tensors: offloading output layer to GPU"]
    targets = {
        **{n: "CUDA_Host" for n in trial["host_tensor_names"]},
        **{n: "CUDA0" for n in trial["gpu_tensor_names"]},
    }
    for name in sorted(targets):
        for _ in range(trial["expected_native_log_occurrences"].get(name, 1)):
            buffer = "CUDA0" if broken == name else targets[name]
            rows.append(f"loader: tensor {name} (10 MiB Q5) "
                        f"buffer type overridden to {buffer}")
    return "\n".join(rows)


def memory(kv=True):
    return {"native_buffer_records": [
        {"category": "kv", "domain": "gpu_vram" if kv else "host_cpu",
         "native_buffer": "CUDA0" if kv else "CPU", "reported_mib": 4.0},
        {"category": "compute", "domain": "gpu_vram",
         "native_buffer": "CUDA0", "reported_mib": 2.0},
    ]}


def probe_result(group_id, peak=5000, prefill=100, decode=50):
    return {
        "mode": group_id, "success": True,
        "native_placement": {"verified": True},
        "gpu_runtime_validation": {"verified": True},
        "measurements": {
            "vram_peak_sampled_mib": peak,
            "host_peak_rss_mib": 1500,
            "prefill_tokens_per_s": prefill,
            "decode_tokens_per_s": decode,
        },
    }


class GroupRankingTests(unittest.TestCase):
    def test_groups_aggregate_omologous_tensors_by_suffix(self):
        groups = group_inventory(layout())
        by_id = {g["id"]: g for g in groups}
        attn = by_id["group:attn_k.weight"]
        self.assertEqual(attn["tensor_count"], 2)
        self.assertEqual(attn["block_indices"], [0, 1])
        self.assertEqual(attn["gguf_unique_weight_bytes"], 20 * MIB)
        self.assertEqual(attn["native_occurrences_expected"], 2)
        self.assertEqual(attn["native_duplicate_uses"], 0)
        self.assertEqual(attn["compute_proxy_ops_per_token"], 4 * 1024**2)
        self.assertTrue(attn["eligible_for_trial"])
        self.assertFalse(by_id["group:attn_norm.weight"]["eligible_for_trial"])
        self.assertEqual(
            by_id["group:attn_norm.weight"]["skip_reason"],
            "below_minimum_memory_benefit",
        )
        self.assertEqual(len(choose_group_trials(groups)), 3)
        self.assertTrue(all(
            row["id"].startswith("group:") for row in choose_group_trials(groups)
        ))

    def test_native_duplicate_counts_not_double_counted_gguf(self):
        for arch in ("gemma2", "llama", "qwen2"):
            with self.subTest(architecture=arch):
                groups = group_inventory(layout(arch))
                ids = {row["id"]: row for row in groups}
                tied = ids["NON_BLOCK:token_embd.weight"]
                self.assertEqual(tied["gguf_unique_weight_bytes"], 300 * MIB)
                if arch == "gemma2":
                    self.assertEqual(tied["native_occurrences_expected"], 2)
                    self.assertEqual(tied["native_duplicate_uses"], 1)
                else:
                    rope = ids["NON_BLOCK:rope_freqs.weight"]
                    self.assertEqual(rope["native_occurrences_expected"], 2)
                    self.assertEqual(rope["gguf_unique_weight_bytes"], 512)
                self.assertFalse(tied["eligible_for_trial"])

    def test_native_patterns_cover_unique_tensors_and_repeated_uses(self):
        for arch in ("gemma2", "llama", "qwen2"):
            with self.subTest(architecture=arch):
                structure = layout(arch)
                groups = group_inventory(structure)
                for choice in [None] + choose_group_trials(groups):
                    trial = native_group_trial(structure, choice)
                    self.assertEqual(
                        trial["host_tensor_count"] + trial["gpu_tensor_count"],
                        len(structure["tensors"]),
                    )
                    self.assertTrue(
                        verify_override_logs(trial, native_log(trial))["verified"]
                    )
                    if choice:
                        self.assertTrue(all(
                            name.endswith(choice["tensor_suffix"])
                            for name in trial["host_tensor_names"]
                        ))

    def test_exact_native_host_verification_rejects_gpu_fallback(self):
        structure = layout()
        trial = native_group_trial(
            structure, choose_group_trials(group_inventory(structure))[0]
        )
        self.assertFalse(
            verify_override_logs(
                trial, native_log(trial, broken=trial["host_tensor_names"][0])
            )["verified"]
        )

    def test_invalid_inventory_and_unknown_shapes_are_not_promoted(self):
        structure = layout()
        structure["tensors"][2]["name"] = structure["tensors"][1]["name"]
        with self.assertRaisesRegex(ValueError, "duplicati"):
            group_inventory(structure)
        structure = layout()
        structure["tensors"][1]["shape"] = None
        groups = {g["id"]: g for g in group_inventory(structure)}
        self.assertFalse(groups["group:attn_k.weight"]["eligible_for_trial"])
        self.assertEqual(
            groups["group:attn_k.weight"]["skip_reason"],
            "incomplete_shape_metadata",
        )


class NativeBenchmarkTests(unittest.TestCase):
    def test_context_is_bounded_below_phase1_maximum(self):
        self.assertEqual(benchmark_context(256), (128, 16))
        self.assertEqual(benchmark_context(7424), (512, 32))

    def test_worker_requires_native_placement_cuda_kv_and_throughput(self):
        structure = layout()
        trial = native_group_trial(
            structure, choose_group_trials(group_inventory(structure))[0]
        )
        worker = {
            "worker_ok": True, "model_id": "qwen-2.5-7b-instruct",
            "mode": trial["id"], "n_ctx": 128, "requested_tokens": 16,
            "measurements": {
                "prompt_tokens": 111, "generated_tokens": 16,
                "vram_peak_sampled_mib": 5000,
                "host_peak_rss_mib": 1500,
                "prefill_tokens_per_s": 100,
                "decode_tokens_per_s": 50,
                "nvml_errors": [],
            },
        }
        with (
            patch("calibration.mapping_phase.subprocess.run",
                  return_value=SimpleNamespace(
                      returncode=0, stdout=json.dumps(worker),
                      stderr=native_log(trial),
                  )),
            patch("calibration.mapping_phase.parse_native_memory",
                  return_value=memory(True)),
        ):
            result = probe_native_mapping(
                Path("/dummy/gguf"), worker["model_id"], trial,
                n_ctx=128, tokens=16,
            )
        self.assertTrue(result["success"])
        self.assertIsNone(result["failure_class"])

        with (
            patch("calibration.mapping_phase.subprocess.run",
                  return_value=SimpleNamespace(
                      returncode=0, stdout=json.dumps(worker),
                      stderr=native_log(trial),
                  )),
            patch("calibration.mapping_phase.parse_native_memory",
                  return_value=memory(False)),
        ):
            result = probe_native_mapping(
                Path("/dummy/gguf"), worker["model_id"], trial,
                n_ctx=128, tokens=16,
            )
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_class"], "gpu_kv_compute_not_verified")

    def test_repeat_medians_reject_regression_and_unstable_runs(self):
        baseline = _median_stats([
            probe_result("baseline", 5000, 100, 50) for _ in range(3)
        ])
        good = _median_stats([
            probe_result("group:attn_k.weight", 4950, 100, 50)
            for _ in range(3)
        ])
        accepted = _compare_with_baseline(baseline, good)
        self.assertEqual(accepted["measured_vram_recovered_mib"], 50)
        self.assertEqual(
            accepted["verdict"], "promising_needs_operational_confirmation"
        )
        slow = _median_stats([
            probe_result("group:attn_k.weight", 4950, 100, 40)
            for _ in range(3)
        ])
        self.assertEqual(
            _compare_with_baseline(baseline, slow)["verdict"],
            "rejected_performance_regression",
        )
        unstable = _median_stats([
            probe_result("group:attn_k.weight", 4950, 100, d)
            for d in (30, 50, 70)
        ])
        self.assertEqual(
            _compare_with_baseline(baseline, unstable)["verdict"],
            "inconclusive_variability",
        )


class MappingEvaluationTests(unittest.TestCase):
    def setup(self, tmp):
        root = Path(tmp)
        path = root / "model.gguf"
        path.write_bytes(b"fixture")
        structure = layout()
        baseline = {
            "model_id": "qwen-2.5-7b-instruct",
            "source": "kaggle", "repository": "qwen-lm/qwen2.5",
            "filename": path.name, "local_path": str(path),
            "architecture": "qwen2", "quantization": "q5_k_m",
            "native_verified_all_gpu": True,
        }
        matrix = {
            "success": True, "status": "enumeration_only",
            "filename": path.name, "model_path": str(path.resolve()),
            "block_count": 2,
            "serialized_weight_bytes": structure["weights_bytes"],
        }
        context = {
            "success": True, "filename": path.name,
            "quantization": "q5_k_m",
            "confirmed_max_gpu_resident_n_ctx": 7424,
        }
        return path, structure, baseline, matrix, context

    def test_runs_repeated_baseline_and_ranked_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, structure, baseline, matrix, context = self.setup(tmp)
            calls = []
            def success(path, model_id, trial, **kwargs):
                calls.append(trial["id"])
                peak = 5000 if trial["id"] == "baseline" else 4950
                return probe_result(trial["id"], peak)
            with (
                patch("calibration.mapping_phase.inspect_gguf",
                      return_value=structure),
                patch("calibration.mapping_phase.gguf_file_set",
                      return_value=[path]),
                patch("calibration.mapping_phase.probe_native_mapping",
                      side_effect=success),
            ):
                report = run_mapping_phase(
                    baseline, matrix, context, model_id=baseline["model_id"]
                )
        self.assertTrue(report["success"])
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["status"],
                         "diagnostic_group_comparison_completed")
        self.assertEqual(calls[:3], ["baseline"] * 3)
        self.assertEqual(len(calls), 12)
        self.assertEqual(len(report["group_evaluations"]), 3)
        self.assertEqual(len(report["promising_group_ids"]), 3)
        self.assertIsNone(report["selected_mapping"])
        self.assertFalse(report["profile_generated"])

    def test_slow_group_rejected_without_failing_complete_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, structure, baseline, matrix, context = self.setup(tmp)
            def slow(path, mid, trial, **kwargs):
                return probe_result(
                    trial["id"],
                    5000 if trial["id"] == "baseline" else 4900,
                    100, 50 if trial["id"] == "baseline" else 30,
                )
            with (
                patch("calibration.mapping_phase.inspect_gguf",
                      return_value=structure),
                patch("calibration.mapping_phase.gguf_file_set",
                      return_value=[path]),
                patch("calibration.mapping_phase.probe_native_mapping",
                      side_effect=slow),
            ):
                report = run_mapping_phase(
                    baseline, matrix, context, model_id=baseline["model_id"]
                )
        self.assertTrue(report["success"])
        self.assertFalse(report["promising_group_ids"])
        self.assertTrue(all(
            row["status"] == "rejected_performance_regression"
            for row in report["group_evaluations"]
        ))

    def test_native_failure_blocks_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, structure, baseline, matrix, context = self.setup(tmp)
            called = []
            def maybe_fail(path, mid, trial, **kwargs):
                called.append(trial["id"])
                if len(called) == 4:
                    return {
                        "mode": trial["id"], "success": False,
                        "failure_class": "worker_protocol_error",
                        "error": "SIGABRT cause unknown",
                    }
                return probe_result(trial["id"])
            with (
                patch("calibration.mapping_phase.inspect_gguf",
                      return_value=structure),
                patch("calibration.mapping_phase.gguf_file_set",
                      return_value=[path]),
                patch("calibration.mapping_phase.probe_native_mapping",
                      side_effect=maybe_fail),
            ):
                report = run_mapping_phase(
                    baseline, matrix, context, model_id=baseline["model_id"]
                )
        self.assertFalse(report["success"])
        self.assertEqual(len(called), 4)
        self.assertEqual(
            report["status"], "blocked_unverified_native_mapping"
        )


if __name__ == "__main__":
    unittest.main()
