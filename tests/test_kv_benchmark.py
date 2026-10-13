"""CPU-only checks for fixed-context KV speed/quality benchmarking."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from calibration.kv_quality_worker import (
    kl_nats, sampled_metrics, score_positions,
)
from calibration.kv_benchmark import (
    eligible_profiles, score_context, run_benchmark,
)


class KvBenchmarkTests(unittest.TestCase):
    def test_complete_realistic_frontier(self):
        matrix = [
            ("off4", 512), ("on_fp16_q8", 512), ("on_q8_fp16", 512),
            ("on_q8_q8", 512), ("on_fp16_q4", 512), ("on_q4_fp16", 512),
            ("on_q8_q4", 768), ("on_q4_q8", 768), ("on_q4_q4", 1280),
        ]
        report = {"profiles": [{
            "id": ident, "status": "confirmed", "max_gpu_n_ctx": maximum
        } for ident, maximum in matrix]}
        self.assertEqual([len(eligible_profiles(report, c))
                          for c in (512, 768, 1280)], [9, 3, 1])
        self.assertEqual(eligible_profiles(report, 1281), [])

    def test_kl_is_zero_for_identical_logits_and_positive_for_shift(self):
        base = np.array([3., 2., 0., -1.], dtype=np.float32)
        changed = np.array([0., 2., 3., -1.], dtype=np.float32)
        self.assertLess(kl_nats(base, base), 1e-10)
        self.assertGreater(kl_nats(base, changed), 0.5)
        data = sampled_metrics(
            np.stack([base, base]), np.stack([changed, changed]),
            [0, 0],
        )
        self.assertEqual(data["samples"], 2)
        self.assertGreater(data["candidate_sampled_ppl"],
                           data["reference_sampled_ppl"])
        self.assertGreater(data["mean_kl_nats"], 0)
        with self.assertRaises(ValueError):
            sampled_metrics(
                np.stack([base]), np.stack([base, base]), [0],
            )

    def test_positions_sparse_sorted_and_have_next_token(self):
        for ctx in (512, 768, 1280):
            positions = score_positions(ctx, 64)
            self.assertEqual(len(positions), 64)
            self.assertEqual(positions, sorted(set(positions)))
            self.assertGreaterEqual(positions[0], 16)
            self.assertLessEqual(positions[-1], ctx - 2)

    def test_score_penalizes_worse_quality_and_tiny_vram_margin(self):
        def row(ident, kl, ppl, speed, peak):
            return {
                "id": ident,
                "speed": {
                    "success": True, "median_prefill_tok_s": speed,
                    "median_decode_tok_s": speed / 10,
                    "max_vram_peak_mib": peak,
                },
                "quality": {"success": True, "quality": {
                    "reference_sampled_ppl": 10.,
                    "candidate_sampled_ppl": ppl,
                    "mean_kl_nats": kl,
                }},
            }
        rows = [
            row("better", 0.001, 10.1, 500, 6080),
            row("worse", 0.025, 12.0, 500, 6138),
        ]
        score_context(
            rows, vram_total_mib=6141,
            kl_scale=0.01, target_headroom_mib=64,
        )
        self.assertGreater(rows[0]["diagnostic_score"],
                           rows[1]["diagnostic_score"])
        self.assertFalse(rows[1]["headroom_target_met"])
        self.assertEqual(rows[1]["vram_headroom_mib"], 3)

    def test_runner_selects_9_3_1_profiles_and_both_quality_controls(self):
        caps = [512] * 6 + [768] * 2 + [1280]
        matrix = []
        for i, cap in enumerate(caps):
            matrix.append({
                "id": f"candidate{i}", "status": "confirmed",
                "flash_attn": i != 0,
                "type_k": "q4_0", "type_v": "f16" if i == 0 else "q4_0",
                "max_gpu_n_ctx": cap,
                "result": {"gpu_total_mib": 6141},
            })
        source = {
            "schema": "ruth-phase1-gpu-kv-profiles-v3",
            "parameters": {"model_id": "gemma-2-9b-it", "source": None},
            "profiles": matrix,
        }
        requests = []
        def fake_speed(model_path, model_id, profile, **kwargs):
            requests.append(("speed", kwargs["n_ctx"], profile["id"]))
            return {
                "success": True, "median_prefill_tok_s": 500.,
                "median_decode_tok_s": 25., "max_vram_peak_mib": 6125,
            }
        def fake_quality(**kwargs):
            requests.append(("quality", kwargs["n_ctx"], kwargs["type_k"],
                             kwargs["reference"] is not None))
            if kwargs["reference"] is None:
                kwargs["output"].touch()
                return {"success": True}
            return {"success": True, "quality": {
                "reference_sampled_ppl": 8.0,
                "candidate_sampled_ppl": 8.1,
                "mean_kl_nats": 0.002,
            }}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            input_json = root / "phase1.json"
            input_json.write_text(json.dumps(source), encoding="utf-8")
            corpus = root / "corpus.txt"
            corpus.write_text("A" * 12000, encoding="utf-8")
            model = root / "model.gguf"
            model.touch()
            with (patch("calibration.kv_benchmark.inspect_gguf",
                        return_value={"metadata": {
                            "general.architecture": "gemma2",
                            "gemma2.context_length": 8192,
                        }}),
                  patch("calibration.kv_benchmark.measure_speed",
                        side_effect=fake_speed),
                  patch("calibration.kv_benchmark.quality_subprocess",
                        side_effect=fake_quality)):
                output = root / "benchmark.json"
                result = run_benchmark(
                    phase1_report=input_json, corpus=corpus,
                    model_path=model, output=output, repetitions=2,
                )
            self.assertTrue(result["completed"])
            self.assertEqual([len(r["profiles"]) for r in result["results"]],
                             [9, 3, 1])
            self.assertEqual(len([r for r in requests if r[0] == "speed"]), 13)
            self.assertTrue(all(
                p["diagnostic_score"] is not None
                for block in result["results"] for p in block["profiles"]
            ))
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
