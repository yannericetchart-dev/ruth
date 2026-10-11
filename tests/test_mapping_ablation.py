"""Offline tests for exhaustive leave-one-out ablation and paired summaries."""

import unittest

from calibration.mapping_ablation import build_ablation_variants, summarize_ablation
from calibration.mapping_candidates import propose_mapping_candidates


SEED = [
    "single:host_to_gpu:blk.0.ffn_gate.weight",
    "single:host_to_gpu:blk.0.attn_k.weight",
]


def fixture():
    rows = [
        {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100},
        {"name": "output_norm.weight", "group": "output", "block": None, "bytes": 8},
    ]
    for block in range(4):
        rows.extend([
            {"name": f"blk.{block}.ffn_gate.weight",
             "group": "feed_forward", "block": block, "bytes": 20},
            {"name": f"blk.{block}.attn_k.weight",
             "group": "attention", "block": block, "bytes": 4},
            {"name": f"blk.{block}.attn_norm.weight",
             "group": "normalization", "block": block, "bytes": 1},
        ])
    return propose_mapping_candidates({
        "metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
        "tensors": rows,
    }, 3)


def observations(variants, runs=3):
    settings = {
        "baseline": (22.0, 280.0, 5909.0),
        "full_seed": (22.4, 285.0, 5941.0),
        "without_0": (22.3, 283.0, 5913.0),
        "without_1": (22.2, 282.0, 5937.0),
    }
    rows = []
    for run in range(1, runs + 1):
        for variant in variants:
            decode, prefill, vram = settings[variant["id"]]
            rows.append({
                "run": run,
                "variant_id": variant["id"],
                "success": True,
                "native_verified": bool(variant["component_ids"]) or None,
                "decode_tps": decode + run / 100,
                "prefill_tps": prefill + run,
                "vram_peak_mib": vram,
                "generated_tokens": 128,
            })
    return rows


class MappingAblationTests(unittest.TestCase):
    def test_builds_baseline_full_and_each_omission(self):
        variants = build_ablation_variants(fixture(), SEED)
        self.assertEqual(
            [v["id"] for v in variants],
            ["baseline", "full_seed", "without_0", "without_1"],
        )
        self.assertEqual(variants[0]["component_ids"], [])
        self.assertEqual(variants[1]["component_ids"], SEED)
        self.assertEqual(variants[2]["component_ids"], SEED[1:])
        self.assertEqual(variants[3]["component_ids"], SEED[:1])
        self.assertEqual(
            [v["removed_component"] for v in variants[2:]], SEED
        )

    def test_rejects_duplicate_invalid_or_too_short_seeds(self):
        proposals = fixture()
        with self.assertRaises(ValueError):
            build_ablation_variants(proposals, SEED[:1])
        with self.assertRaises(ValueError):
            build_ablation_variants(proposals, SEED + [SEED[0]])
        with self.assertRaises(ValueError):
            build_ablation_variants(proposals, SEED + ["missing-id"])

    def test_paired_attribution_and_vram(self):
        variants = build_ablation_variants(fixture(), SEED)
        report = summarize_ablation(observations(variants), variants, 3, 6141)
        self.assertTrue(report["descriptive_only"])
        self.assertEqual(len(report["variants"]), 4)
        full = report["variants"][1]
        self.assertEqual(full["decode_faster_than_baseline_pairs"], 3)
        self.assertEqual(full["sampled_headroom_at_peak_mib"], 200)
        self.assertEqual(full["paired_decode_vs_baseline_tps"], [0.4] * 3)
        marginal = report["component_marginal_contributions"]
        self.assertEqual(marginal[0]["component_id"], SEED[0])
        self.assertEqual(marginal[0]["median_marginal_decode_tps"], 0.1)
        self.assertEqual(marginal[1]["median_marginal_decode_tps"], 0.2)
        self.assertEqual(marginal[0]["full_seed_decode_faster_pairs"], 3)
        self.assertEqual(marginal[0]["marginal_vram_mib_by_round"], [28] * 3)
        self.assertAlmostEqual(
            report["two_component_interaction"]["median_decode_nonadditivity_tps"],
            -0.1,
        )

    def test_reject_missing_and_duplicate_pairs(self):
        variants = build_ablation_variants(fixture(), SEED)
        data = observations(variants)
        with self.assertRaises(ValueError):
            summarize_ablation(data[:-1], variants, 3, 6141)
        with self.assertRaises(ValueError):
            summarize_ablation(data + [data[0]], variants, 3, 6141)

    def test_reject_failed_or_unverified_native_mapping(self):
        variants = build_ablation_variants(fixture(), SEED)
        data = observations(variants)
        data[1]["native_verified"] = False
        with self.assertRaises(ValueError):
            summarize_ablation(data, variants, 3, 6141)
        data = observations(variants)
        data[2]["success"] = False
        with self.assertRaises(ValueError):
            summarize_ablation(data, variants, 3, 6141)

    def test_reject_short_nonfinite_and_negative_metrics(self):
        variants = build_ablation_variants(fixture(), SEED)
        data = observations(variants)
        data[1]["generated_tokens"] = 9
        with self.assertRaises(ValueError):
            summarize_ablation(data, variants, 3, 6141)
        data = observations(variants)
        data[1]["decode_tps"] = float("inf")
        with self.assertRaises(ValueError):
            summarize_ablation(data, variants, 3, 6141)
        data = observations(variants)
        data[1]["vram_peak_mib"] = -1
        with self.assertRaises(ValueError):
            summarize_ablation(data, variants, 3, 6141)


if __name__ == "__main__":
    unittest.main()
