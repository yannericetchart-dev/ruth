"""Offline tests for speed-only post-optimization of confirmed GPU-KV maps."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.mapping_grouping import native_tensor_trial
from calibration.kv_post_optimize import (
    candidate_pool, neighbors, objective, pareto_front,
    run_post_optimize, _valid_seed,
)
from tests.test_kv_adaptive_tensor_offload import layout_fixture


def verified_seed(names, mapping):
    return {
        "success": True, "n_ctx": 512, "repetitions": 3,
        "host_tensor_names": sorted(names),
        "mapping_id": mapping,
        "metrics": {
            "prefill_tok_s": 400., "decode_tok_s": 20.,
            "vram_peak_mib": 6120., "vram_total_mib": 6141.,
        },
        "samples": [{
            "success": True, "native_placement_verified": True,
            "gpu_kv_verified": True, "native_kv_type_verified": True,
        } for _ in range(3)],
    }


def prepare_source(root):
    model = root / "model.gguf"
    model.write_bytes(b"fake gguf")
    seed_names = ["blk.0.ffn_down.weight"]
    seed = verified_seed(seed_names, native_tensor_trial(
        layout_fixture(), seed_names
    )["id"])
    report = {
        "schema": "ruth-phase2-adaptive-tensor-greedy-v3",
        "completed": True,
        "parameters": {
            "model_id": "gemma-2-9b-it",
            "model_path": str(model.resolve()),
            "generated_tokens": 16, "confirm_repeats": 3,
        },
        "stages": [{
            "successful": True, "measured_target_n_ctx": 512,
            "confirmed_host_tensor_names": seed_names, "validation": seed,
            "individual_screenings": [],
        }],
        "learned_groups": [{
            "target_n_ctx": 512, "cumulative_host_tensor_names": seed_names,
        }],
        "measured_configs": {},
    }
    path = root / "greedy.json"
    path.write_text(json.dumps(report))
    return path, seed


def fake_gpu_run(path, model_id, trial, *,
                 n_ctx, tokens, timeout, flash_attn, type_k, type_v):
    names = set(trial["host_tensor_names"])
    valid = bool(names)
    d = 20.
    p = 400.
    vram = 6120.
    if "blk.0.attn_k.weight" in names:
        # Better decode, *less* available VRAM (2 MiB headroom).
        d = 24.
        p = 390.
        vram = 6139.
    if "blk.1.attn_k.weight" in names:
        d, p = 18., 380.
    return {
        "success": valid, "failure_class":
            None if valid else "native_cuda_oom",
        "native_placement": {"verified": valid},
        "gpu_runtime_validation": {"verified": valid},
        "kv_config_native_verification": {"verified": valid},
        "measurements": {
            "prefill_tokens_per_s": p, "decode_tokens_per_s": d,
            "vram_peak_sampled_mib": vram,
            "vram_total_mib": 6141.,
            "prompt_tokens": n_ctx-tokens,
            "generated_tokens": tokens, "total_tokens": n_ctx,
            "host_peak_rss_mib": 250.,
        },
    }


class PostOptimizeTests(unittest.TestCase):
    def test_speed_objective_ignores_vram_and_pareto_uses_only_speeds(self):
        slow = {"prefill_tok_s": 400., "decode_tok_s": 20.,
                "vram_peak_mib": 6100.}
        fast = {"prefill_tok_s": 390., "decode_tok_s": 24.,
                "vram_peak_mib": 6139.}
        self.assertGreater(objective(fast), objective(slow))
        self.assertEqual(objective({**fast, "vram_peak_mib": 6139}),
                         objective({**fast, "vram_peak_mib": 100}))
        rows = [
            {"success": True, "metrics": {
                "prefill_tok_s": 400., "decode_tok_s": 20.}},
            {"success": True, "metrics": {
                "prefill_tok_s": 390., "decode_tok_s": 24.}},
            {"success": True, "metrics": {
                "prefill_tok_s": 380., "decode_tok_s": 19.}},
        ]
        self.assertEqual(len(pareto_front(rows)), 2)

    def test_neighbor_generator_remove_replace_add(self):
        rows = list(neighbors(
            ["blk.0.ffn_down.weight"], ["blk.0.attn_k.weight"]
        ))
        self.assertEqual(
            [r["operation"] for r in rows], ["remove", "swap", "add"]
        )
        self.assertEqual(
            rows[1]["host_tensor_names"], ["blk.0.attn_k.weight"]
        )

    def test_missing_native_confirmation_cannot_be_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            report, seed = prepare_source(Path(folder))
            source = json.loads(report.read_text())
            bad = {**seed, "samples": [{**s} for s in seed["samples"]]}
            bad["samples"][0]["gpu_kv_verified"] = False
            with self.assertRaisesRegex(ValueError, "verifica nativa"):
                _valid_seed(source, {
                    **source["stages"][0], "validation": bad,
                }, target=512, confirmation_runs=3)

    def test_can_choose_faster_candidate_with_only_two_mib_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_path, seed = prepare_source(root)
            output = root / "post.json"
            with (
                patch("calibration.kv_post_optimize.inspect_gguf",
                      return_value=layout_fixture()),
                patch("calibration.kv_post_optimize.candidate_pool",
                      return_value=["blk.0.attn_k.weight",
                                    "blk.1.attn_k.weight"]),
            ):
                result = run_post_optimize(
                    source_report=source_path, output=output,
                    max_tests=20, max_rounds=1,
                    max_candidates=2, probe=fake_gpu_run,
                )
            self.assertTrue(result["completed"])
            self.assertEqual(
                result["best_host_tensor_names"],
                ["blk.0.attn_k.weight"],
            )
            self.assertEqual(result["best"]["metrics"]["vram_peak_mib"],
                             6139.)
            self.assertIsNone(
                result["objective"]["vram_minimum_headroom_mib"]
            )
            self.assertEqual(result["objective"]["vram_objective_weight"], 0)
            self.assertTrue(result["pareto_front"])
            self.assertEqual(result["best"]["repetitions"], 3)

    def test_resume_with_greater_budget_no_repeated_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, seed = prepare_source(root)
            output = root / "checkpoint.json"
            calls = []
            def counted(*args, **kwargs):
                calls.append(tuple(args[2]["host_tensor_names"]))
                return fake_gpu_run(*args, **kwargs)
            with (
                patch("calibration.kv_post_optimize.inspect_gguf",
                      return_value=layout_fixture()),
                patch("calibration.kv_post_optimize.candidate_pool",
                      return_value=["blk.0.attn_k.weight",
                                    "blk.1.attn_k.weight"]),
            ):
                first = run_post_optimize(
                    source_report=source, output=output,
                    max_tests=1, max_rounds=1, max_candidates=2,
                    probe=counted,
                )
                self.assertFalse(first["completed"])
                assert first["status"] == "budget_exhausted_best_valid"
                prior_calls = len(calls)
                second = run_post_optimize(
                    source_report=source, output=output,
                    max_tests=20, max_rounds=1, max_candidates=2,
                    probe=counted,
                )
            self.assertTrue(second["completed"])
            self.assertGreater(len(calls), prior_calls)
            self.assertEqual(len(second["measured_configs"]), 5)
            self.assertEqual(
                calls.count(("blk.0.attn_k.weight",)), 3,
            )


if __name__ == "__main__":
    unittest.main()
