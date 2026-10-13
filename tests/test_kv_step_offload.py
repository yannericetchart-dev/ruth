"""Offline checks for KV step native evidence and multi-group 512 recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.kv_memory_steps import (
    kv_bytes_mib, changes_at_boundary, run_memory_steps,
)
from calibration.kv_step_offload import (
    blocked_profiles, enumerate_group_candidates, speed_trial,
)
from calibration.mapping_grouping import (
    group_inventory, native_multi_group_trial,
)
from calibration.mapping_native_validation import verify_override_logs


def gguf_fixture():
    M = 1048576
    tensors = [
        {"name": "token_embd.weight", "bytes": 20*M,
         "shape": [200, 400], "type": "F16"}
    ]
    for block in range(2):
        for suffix, mib in (("attn_k.weight", 25),
                            ("ffn_down.weight", 45),
                            ("attn_norm.weight", 1)):
            tensors.append({
                "name": f"blk.{block}.{suffix}",
                "bytes": mib*M,
                "shape": [1024, 1024] if "norm" not in suffix else [1024],
                "type": "Q4_K_M",
            })
    return {
        "metadata": {
            "general.architecture": "gemma2",
            "gemma2.block_count": 2,
            "gemma2.context_length": 8192,
        },
        "tensors": tensors,
    }


def native_log(trial):
    lines = ["llama_model_load_tensors: offloading output layer to GPU"]
    for name in trial["host_tensor_names"]:
        lines.append(
            f"loader: tensor {name} (1 MiB Q4) buffer type overridden to CUDA_Host"
        )
    for name in trial["gpu_tensor_names"]:
        count = trial["expected_native_log_occurrences"].get(name, 1)
        for _ in range(count):
            lines.append(
                f"loader: tensor {name} (1 MiB Q4) buffer type overridden to CUDA0"
            )
    return "\n".join(lines)


class NativeMemoryAndMappingTests(unittest.TestCase):
    def test_boundary_crossing_is_observed_not_assumed(self):
        readings = {"255": 9.0, "256": 9.0, "257": 14.25}
        step = changes_at_boundary(readings, 256)
        self.assertTrue(step["native_step_observed"])
        self.assertEqual(step["crossing_step_mib"], 5.25)
        self.assertIsNone(step["gpu_kv_increment_mib"])
        self.assertFalse(changes_at_boundary({
            "511": 20., "512": 20., "513": 20.,
        }, 512)["native_step_observed"])

    def test_native_memory_retains_kv_component_types_and_bytes(self):
        from calibration.native_memory import parse_native_memory
        stderr = (
            "llama_kv_cache: CUDA0 KV buffer size = 118.12 MiB\\n"
            "llama_kv_cache: size = 118.12 MiB, K (q8_0): 70.00 MiB, "
            "V (q4_0): 48.12 MiB\\n"
        )
        native = parse_native_memory(stderr)
        self.assertEqual(native["gpu_model_buffer_mib"], 0)
        self.assertEqual(kv_bytes_mib(native, domain="gpu_vram"), 118.12)
        self.assertEqual(native["native_kv_component_records"], [{
            "type_k": "q8_0", "type_v": "q4_0",
            "k_mib": 70., "v_mib": 48.12,
        }])

    def test_native_host_bytes_require_actual_buffer_records(self):
        self.assertIsNone(kv_bytes_mib({}, domain="host_cpu"))
        records = {"native_buffer_records": [
            {"category": "kv", "domain": "host_cpu", "reported_mib": 13.5},
            {"category": "kv", "domain": "host_cpu", "reported_mib": 8.0},
            {"category": "compute", "domain": "host_cpu", "reported_mib": 50},
        ]}
        self.assertEqual(kv_bytes_mib(records, domain="host_cpu"), 21.5)

    def test_multi_group_native_rules_are_complete_and_disjoint(self):
        layout = gguf_fixture()
        groups = {g["id"]: g for g in group_inventory(layout)}
        trial = native_multi_group_trial(layout, [
            groups["group:attn_k.weight"], groups["group:attn_norm.weight"],
        ])
        self.assertEqual(len(trial["host_tensor_names"]), 4)
        self.assertEqual(trial["host_tensor_count"], 4)
        self.assertEqual(
            trial["group_ids"],
            ["group:attn_k.weight", "group:attn_norm.weight"],
        )
        self.assertTrue(
            verify_override_logs(trial, native_log(trial))["verified"]
        )
        with self.assertRaises(ValueError):
            native_multi_group_trial(layout, [
                groups["group:attn_k.weight"], groups["group:attn_k.weight"],
            ])

    def test_blocked_profile_258_failure_not_labeled_as_confirmed_256(self):
        report = {"profiles": [
            {"id": "ref", "status": "reference_reused",
             "max_gpu_n_ctx": 257},
            {"id": "failed", "status": "unsupported_or_inconclusive",
             "result": {"attempts": [{
                 "requested_n_ctx": 258,
                 "success": False,
                 "failure_class": "native_cuda_oom",
             }]}},
            {"id": "ambiguous", "status": "unsupported_or_inconclusive",
             "result": {"attempts": [{
                 "requested_n_ctx": 258,
                 "success": False,
                 "failure_class": "worker_protocol_error",
             }]}},
            {"id": "native_crash", "status": "unsupported_or_inconclusive",
             "result": {"attempts": [{
                 "requested_n_ctx": 258,
                 "success": False,
                 "failure_class": "worker_protocol_error",
                 "worker_signal": "SIGABRT",
             }]}},
            {"id": "already_512", "status": "confirmed",
             "max_gpu_n_ctx": 512},
        ]}
        self.assertEqual(
            [p["id"] for p in blocked_profiles(report)],
            ["ref", "failed", "native_crash"],
        )

    def test_combinations_ranked_using_proxy_never_prove_vram(self):
        groups = group_inventory(gguf_fixture())
        picked = enumerate_group_candidates(
            groups, proxy_mib=70, safety_mib=8,
            max_groups=2, max_candidates=5,
        )
        self.assertTrue(picked)
        self.assertTrue(all(
            sum(g["gguf_unique_weight_mib"] for g in combo) >= 78
            for combo in picked
        ))
        self.assertTrue(all(len(combo) <= 2 for combo in picked))
        self.assertLessEqual(len(picked), 5)

    def test_multi_group_speed_requires_independent_repeats(self):
        layout = gguf_fixture()
        groups = [g for g in group_inventory(layout)
                  if g["eligible_for_trial"]]
        trial = native_multi_group_trial(layout, groups[:2])
        called = []
        def fake_probe(path, model_id, t, **kw):
            called.append(kw["n_ctx"])
            return {
                "success": True,
                "native_placement": {"verified": True},
                "gpu_runtime_validation": {"verified": True},
                "kv_config_native_verification": {"verified": True},
                "measurements": {
                    "prefill_tokens_per_s": 150.,
                    "decode_tokens_per_s": 12.,
                    "vram_peak_sampled_mib": 6050.,
                },
            }
        with patch("calibration.kv_step_offload.probe_native_mapping",
                   side_effect=fake_probe):
            result = speed_trial(
                Path("/fake.gguf"), "gemma-2-9b-it", trial,
                n_ctx=512, tokens=16, timeout=100, repetitions=3,
                flash_attn=True, type_k="q8_0", type_v="f16",
            )
        self.assertTrue(result["success"])
        self.assertEqual(called, [512, 512, 512])
        self.assertEqual(result["native_group_ids"], trial["group_ids"])
        self.assertEqual(result["native_host_tensors"], trial["host_tensor_names"])

    def test_step_report_handles_reference_without_result(self):
        source = {
            "schema": "ruth-phase1-gpu-kv-profiles-v3",
            "parameters": {"model_id": "gemma-2-9b-it", "source": None},
            "profiles": [{
                "id": "fa_off_k_f16_v_f16", "status": "reference_reused",
                "flash_attn": False, "type_k": "f16", "type_v": "f16",
                "result": None,
            }],
        }
        def fake_probe(path, model_id, trial, **kwargs):
            ctx = kwargs["n_ctx"]
            expected = 30.0 if ctx <= 256 else 50.0
            if ctx > 512:
                expected += 20.0
            return {
                "success": True, "native_memory": {
                    "native_buffer_records": [{
                        "category": "kv", "domain": "host_cpu",
                        "reported_mib": expected,
                    }]
                },
            }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            phase1 = root / "phase1.json"
            phase1.write_text(json.dumps(source))
            model = root / "model.gguf"
            model.touch()
            report = root / "steps.json"
            with (patch("calibration.kv_memory_steps.inspect_gguf",
                        return_value=gguf_fixture()),
                  patch("calibration.kv_memory_steps.native_group_trial",
                        return_value={"id": "baseline"})):
                result = run_memory_steps(
                    phase1_report=phase1, model_path=model,
                    output=report, boundaries=(256, 512),
                    probe=fake_probe,
                )
            self.assertTrue(result["completed"])
            self.assertEqual(result["profiles"][0][
                "additional_kv_mib_257_to_512_host_proxy"
            ], 0.)
            self.assertEqual(
                result["profiles"][0]["steps"][0]["crossing_step_mib"],
                20.0,
            )
            self.assertEqual(
                result["profiles"][0]["steps"][1]["crossing_step_mib"],
                20.0,
            )
            self.assertTrue(report.exists())


    def test_complete_recovery_comparison_uses_existing_full_gpu_rows(self):
        import hashlib
        from calibration.kv_step_offload import run_step_offload
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            corpus.write_text("Testo di valutazione " * 500)
            model = root / "model.gguf"
            model.touch()
            sha_corpus = hashlib.sha256(corpus.read_bytes()).hexdigest()
            p1 = {
                "schema": "ruth-phase1-gpu-kv-profiles-v3",
                "parameters": {"model_id": "gemma-2-9b-it"},
                "profiles": [
                    {"id": "fa_off_k_f16_v_f16",
                     "flash_attn": False, "type_k": "f16", "type_v": "f16",
                     "max_gpu_n_ctx": 257, "status": "reference_reused",
                     "result": None},
                    {"id": "fa_on_k_q8_0_v_q8_0",
                     "flash_attn": True, "type_k": "q8_0", "type_v": "q8_0",
                     "max_gpu_n_ctx": 512, "status": "confirmed",
                     "result": {"gpu_total_mib": 6141}},
                ],
            }
            f1 = root / "phase1.json"
            f1.write_text(json.dumps(p1))
            sha = hashlib.sha256(f1.read_bytes()).hexdigest()
            steps = {
                "schema": "ruth-kv-native-memory-steps-v1",
                "source_phase1_sha256": sha,
                "model_id": "gemma-2-9b-it",
                "model_path": str(model.resolve()),
                "profiles": [{
                    "id": "fa_off_k_f16_v_f16",
                    "additional_kv_mib_257_to_512_host_proxy": 0.,
                }],
            }
            fm = root / "steps.json"
            fm.write_text(json.dumps(steps))
            previous = {
                "schema": "ruth-kv-fixed-quality-speed-v1",
                "source_phase1_sha256": sha, "corpus_sha256": sha_corpus,
                "model_id": "gemma-2-9b-it",
                "model_path": str(model.resolve()),
                "contexts": [512], "kl_scale_nats": 0.01,
                "headroom_target_mib": 64.,
                "results": [{
                    "n_ctx": 512, "profiles": [{
                        "id": "fa_on_k_q8_0_v_q8_0",
                        "diagnostic_score": 90.,
                        "speed": {
                            "success": True, "median_prefill_tok_s": 400.,
                            "median_decode_tok_s": 20.,
                            "max_vram_peak_mib": 6050.,
                        },
                        "quality": {"success": True, "quality": {
                            "reference_sampled_ppl": 8.,
                            "candidate_sampled_ppl": 8.1,
                            "mean_kl_nats": 0.001,
                        }},
                    }],
                }],
            }
            fb = root / "benchmark.json"
            fb.write_text(json.dumps(previous))
            observed = []
            def speed(*args, **kwargs):
                observed.append((kwargs["n_ctx"], kwargs["repetitions"],
                                 args[2]["group_ids"]))
                return {
                    "success": True, "median_prefill_tok_s": 300.,
                    "median_decode_tok_s": 18.,
                    "max_vram_peak_mib": 6100.,
                    "repetitions": [
                        {"success": True} for _ in range(kwargs["repetitions"])
                    ],
                }
            def quality(**kwargs):
                if kwargs["reference"] is None:
                    kwargs["output"].touch()
                    return {"success": True}
                return {"success": True, "quality": {
                    "reference_sampled_ppl": 8.,
                    "candidate_sampled_ppl": 8.05,
                    "mean_kl_nats": 0.0009,
                }}
            with (patch("calibration.kv_step_offload.inspect_gguf",
                        return_value=gguf_fixture()),
                  patch("calibration.kv_step_offload.probe_context",
                        return_value={"success": True, "measurements": {
                            "vram_peak_sampled_mib": 6130.,
                        }}),
                  patch("calibration.kv_step_offload.probe_native_mapping",
                        return_value={"success": True, "measurements": {
                            "vram_peak_sampled_mib": 6000.,
                        }}),
                  patch("calibration.kv_step_offload.speed_trial",
                        side_effect=speed),
                  patch("calibration.kv_step_offload.quality_subprocess",
                        side_effect=quality)):
                output = root / "result.json"
                result = run_step_offload(
                    phase1_report=f1, memory_steps_report=fm,
                    benchmark_report=fb, corpus=corpus,
                    output=output, max_groups=1,
                    max_candidates=1, max_validated=1, repetitions=3,
                )
            self.assertTrue(result["completed"])
            self.assertTrue(result["profiles"][0]["success"])
            self.assertEqual(observed[0][:2], (512, 3))
            candidate = result["profiles"][0]["candidates"][0]
            self.assertEqual(
                candidate["preflight_same_ctx"][
                    "net_vram_peak_saved_vs_full_gpu_mib"
                ], 130.,
            )
            self.assertIsNotNone(candidate["diagnostic_score"])
            self.assertTrue(candidate["host_tensor_names"])
            self.assertEqual(
                len(result["profiles"][0]["normalized_full_gpu_scores"]), 1
            )
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
