"""Offline regression tests for Ruth's adaptive family/block/single-tensor search."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.mapping_grouping import (
    native_tensor_trial, group_inventory,
)
from calibration.mapping_native_validation import verify_override_logs
from calibration.kv_adaptive_tensor_offload import (
    discovery_groups, marginal_value, run_adaptive_tensor_search,
)


MIB = 1048576


def layout_fixture():
    tensors = [{
        "name": "token_embd.weight",
        "bytes": 2*MIB, "shape": [128, 128], "type": "Q4",
    }]
    for block in range(2):
        for suffix, size in (
            ("attn_k.weight", 15), ("ffn_down.weight", 60),
            ("attn_norm.weight", 1),
        ):
            tensors.append({
                "name": f"blk.{block}.{suffix}",
                "bytes": size*MIB,
                "shape": [128, 128] if "norm" not in suffix else [128],
                "type": "Q4",
            })
    return {
        "metadata": {
            "general.architecture": "gemma2",
            "gemma2.block_count": 2,
            "gemma2.context_length": 8192,
        },
        "tensors": tensors,
    }


def native_evidence(trial):
    rows = ["llama_model_load_tensors: offloading output layer to GPU"]
    for name in sorted(trial["host_tensor_names"] + trial["gpu_tensor_names"]):
        device = "CUDA_Host" if name in trial["host_tensor_names"] else "CUDA0"
        expected = trial["expected_native_log_occurrences"].get(name, 1)
        for _ in range(expected):
            rows.append(
                f"tensor {name} (5 MiB Q4) buffer type overridden to {device}"
            )
    return "\n".join(rows)


def prepare_reports(root):
    p1 = {
        "schema": "ruth-phase1-gpu-kv-profiles-v3",
        "parameters": {"model_id": "gemma-2-9b-it"},
        "profiles": [{
            "id": "fa_off_k_f16_v_f16", "flash_attn": False,
            "type_k": "f16", "type_v": "f16",
            "status": "reference_reused", "max_gpu_n_ctx": 257,
            "result": None,
        }],
    }
    phase_path = root / "phase1.json"
    phase_path.write_text(json.dumps(p1))
    steps = {
        "schema": "ruth-kv-native-memory-steps-v1",
        "source_phase1_sha256":
            hashlib.sha256(phase_path.read_bytes()).hexdigest(),
        "model_id": "gemma-2-9b-it",
        "profiles": [{
            "id": "fa_off_k_f16_v_f16",
            "observed_edge_gcd_tokens": 256,
            "steps": [
                {"boundary_tokens": 256, "crossing_step_mib": 12.0},
                {"boundary_tokens": 512, "crossing_step_mib": 12.0},
            ],
            "additional_kv_mib_257_to_512_host_proxy": 0.0,
        }],
    }
    steps_path = root / "steps.json"
    steps_path.write_text(json.dumps(steps))
    model = root / "model.gguf"
    model.write_bytes(b"GGUF simulated")
    return model, phase_path, steps_path


def fake_native_probe(model_path, model_id, trial, *,
                      n_ctx, tokens, timeout, flash_attn, type_k, type_v):
    names = trial["host_tensor_names"]
    gains = sum(
        6 if "norm" in name else
        15 if "attn_k" in name else 40 for name in names
    )
    prefill_cost = sum(
        0.05 if "norm" in name else
        8 if "attn_k" in name else 100 for name in names
    )
    decode_cost = sum(
        0.01 if "norm" in name else
        0.1 if "attn_k" in name else 4 for name in names
    )
    success = (
        n_ctx <= 257 or
        (n_ctx <= 512 and gains >= 6) or
        (n_ctx <= 768 and gains >= 12)
    )
    return {
        "success": success,
        "failure_class": None if success else "native_cuda_oom",
        "native_placement": {"verified": success},
        "gpu_runtime_validation": {"verified": success},
        "kv_config_native_verification": {"verified": success},
        "measurements": {
            "prompt_tokens": n_ctx - tokens,
            "generated_tokens": tokens,
            "total_tokens": n_ctx,
            "prefill_tokens_per_s": 500. - prefill_cost,
            "decode_tokens_per_s": 25. - decode_cost,
            "vram_peak_sampled_mib": 6100. - gains,
            "vram_total_mib": 6141.,
            "host_peak_rss_mib": 500.,
        },
    }


class NativeTensorMappingTests(unittest.TestCase):
    def test_split_same_suffix_across_different_blocks(self):
        fixture = layout_fixture()
        name = "blk.0.ffn_down.weight"
        trial = native_tensor_trial(fixture, [name])
        self.assertEqual(trial["host_tensor_names"], [name])
        self.assertIn("blk.1.ffn_down.weight", trial["gpu_tensor_names"])
        self.assertTrue(
            verify_override_logs(trial, native_evidence(trial))["verified"]
        )
        self.assertTrue(any(
            rule["pattern"] == r"^blk\.0\.ffn_down\.weight$"
            and rule["device"] == "cpu" for rule in trial["rules"]
        ))
        self.assertTrue(any(
            rule["pattern"] == r"^blk\.1\.ffn_down\.weight$"
            and rule["device"] == "gpu" for rule in trial["rules"]
        ))

    def test_reject_duplicate_unknown_and_tied_output(self):
        fixture = layout_fixture()
        cases = [
            ["blk.0.attn_k.weight", "blk.0.attn_k.weight"],
            ["blk.99.attn_k.weight"],
            ["token_embd.weight"],
            ["output.weight"],
        ]
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                native_tensor_trial(fixture, bad)
        self.assertEqual(native_tensor_trial(fixture, [])["host_tensor_count"], 0)

    def test_group_candidates_include_both_full_blocks_and_families(self):
        groups = discovery_groups(layout_fixture())
        by_id = {g["id"]: g for g in groups}
        self.assertEqual(len(by_id["group:attn_k.weight"]["tensor_names"]), 2)
        self.assertEqual(len(by_id["block:0"]["tensor_names"]), 3)
        self.assertEqual(len(by_id["block:1"]["tensor_names"]), 3)
        self.assertTrue(all(g["kind"] in ("family", "block") for g in groups))


class GreedyLogicTests(unittest.TestCase):
    def test_marginal_efficiency_uses_actual_speed_and_peak(self):
        a = {"success": True, "metrics": {
            "prefill_tok_s": 500., "decode_tok_s": 25.,
            "vram_peak_mib": 6100.,
        }}
        b = {"success": True, "metrics": {
            "prefill_tok_s": 490., "decode_tok_s": 24.,
            "vram_peak_mib": 6080.,
        }}
        ranking = marginal_value(a, b)
        self.assertTrue(ranking["eligible"])
        self.assertEqual(ranking["gain_mib"], 20.)
        self.assertGreater(ranking["prefill_loss_pct"], 0)
        self.assertGreater(ranking["decode_loss_pct"], 0)
        self.assertGreater(ranking["efficiency"], 0)
        reverse = marginal_value(b, a)
        self.assertFalse(reverse["eligible"])

    def test_learning_two_successive_context_steps_and_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, phase, steps = prepare_reports(root)
            target = root / "greedy.json"
            calls = []
            def probe(*args, **kwargs):
                calls.append((kwargs["n_ctx"],
                              tuple(args[2]["host_tensor_names"])))
                return fake_native_probe(*args, **kwargs)
            with patch("calibration.kv_adaptive_tensor_offload.inspect_gguf",
                       return_value=layout_fixture()):
                result = run_adaptive_tensor_search(
                    model_path=model, phase1_report=phase,
                    memory_steps_report=steps,
                    targets=(512, 768), max_tests=250, max_frozen=4,
                    max_backtracks=1, output=target, probe=probe,
                )
                self.assertTrue(result["completed"])
                self.assertEqual(len(result["learned_groups"]), 2)
                self.assertEqual(
                    [g["target_n_ctx"] for g in result["learned_groups"]],
                    [512, 768],
                )
                self.assertEqual(
                    [len(g["cumulative_host_tensor_names"])
                     for g in result["learned_groups"]], [1, 2],
                )
                self.assertEqual(
                    [s["status"] for s in result["stages"]],
                    ["confirmed_gpu_resident", "confirmed_gpu_resident"],
                )
                self.assertTrue(all(
                    s["validation"]["repetitions"] == 3
                    for s in result["stages"]
                ))
                self.assertTrue(all(
                    s["group_screenings"] and s["individual_screenings"]
                    for s in result["stages"]
                ))
                snapshot_count = len(calls)
                resumed = run_adaptive_tensor_search(
                    model_path=model, phase1_report=phase,
                    memory_steps_report=steps,
                    targets=(512, 768), max_tests=250, max_frozen=4,
                    max_backtracks=1, output=target, probe=probe,
                )
                self.assertTrue(resumed["completed"])
                self.assertEqual(len(calls), snapshot_count)

    def test_insufficient_probe_budget_never_promotes_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, phase, steps = prepare_reports(root)
            with patch("calibration.kv_adaptive_tensor_offload.inspect_gguf",
                       return_value=layout_fixture()):
                result = run_adaptive_tensor_search(
                    model_path=model, phase1_report=phase,
                    memory_steps_report=steps,
                    targets=(512,), max_tests=1,
                    output=root / "insufficient.json", probe=fake_native_probe,
                )
            self.assertFalse(result["completed"])
            self.assertEqual(result["learned_groups"], [])
            self.assertEqual(result["final_host_tensor_names"], [])
            self.assertTrue(result["stages"][0]["tentative_moves_not_promoted"])


if __name__ == "__main__":
    unittest.main()
