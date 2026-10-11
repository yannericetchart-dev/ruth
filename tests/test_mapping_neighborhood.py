"""Pure/offline coverage for the local, measurement-first neighborhood search."""

import unittest

from calibration.mapping_candidates import propose_mapping_candidates
from calibration.mapping_composition import compose_candidates
from calibration.mapping_neighborhood import choose_neighbors, summarize_neighborhood


def proposals():
    records = [
        {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 200},
        {"name": "output_norm.weight", "group": "output", "block": None, "bytes": 4},
    ]
    for block in range(5):
        records += [
            {"name": f"blk.{block}.attn_k.weight", "group": "attention",
             "block": block, "bytes": 5},
            {"name": f"blk.{block}.attn_q.weight", "group": "attention",
             "block": block, "bytes": 9},
            {"name": f"blk.{block}.ffn_gate.weight", "group": "feed_forward",
             "block": block, "bytes": 20},
            {"name": f"blk.{block}.ffn_down.weight", "group": "feed_forward",
             "block": block, "bytes": 25},
            {"name": f"blk.{block}.attn_norm.weight", "group": "normalization",
             "block": block, "bytes": 1},
        ]
    return propose_mapping_candidates({
        "metadata": {"general.architecture": "gemma2", "gemma2.block_count": 5},
        "tensors": records,
    }, 4)


SEED = [
    "single:host_to_gpu:blk.0.attn_k.weight",
    "single:host_to_gpu:blk.0.ffn_gate.weight",
]


class NeighborhoodTests(unittest.TestCase):
    def test_selects_all_available_families_without_hardcoded_allowlist(self):
        proposed = proposals()
        neighbors = choose_neighbors(proposed, SEED, 3)
        self.assertEqual(
            {item["family"] for item in neighbors},
            {"attention", "feed_forward", "normalization"},
        )
        self.assertEqual(len({item["id"] for item in neighbors}), 3)
        seed_names = set(compose_candidates(proposed, SEED)["tensor_names"])
        for item in neighbors:
            self.assertTrue(seed_names.isdisjoint(item["tensor_names"]))
            self.assertFalse(item["tensor_names"][0].startswith("blk.0."))
            composed = compose_candidates(proposed, SEED + [item["id"]])
            self.assertEqual(len(composed["tensor_names"]), 3)

    def test_candidate_budget_and_repeatability(self):
        proposed = proposals()
        a = choose_neighbors(proposed, SEED, 4)
        b = choose_neighbors(proposed, SEED, 4)
        self.assertEqual([c["id"] for c in a], [c["id"] for c in b])
        self.assertEqual(len(a), 4)
        with self.assertRaises(ValueError):
            choose_neighbors(proposed, SEED, 0)

    @staticmethod
    def synthetic_trials(neighbors, runs=3):
        rows = []
        ids = ["seed"] + [c["id"] for c in neighbors]
        for run in range(1, runs + 1):
            for index, id_ in enumerate(ids):
                rows.append({
                    "run": run, "id": id_,
                    "success": True, "mapping_verified": True,
                    "decode_tps": 22.3 + run * 0.01 + index * 0.05,
                    "prefill_tps": 280 + run + index,
                    "vram_peak_mib": 5941 + index * 4,
                    "generated_tokens": 128,
                })
        return rows

    def test_summary_computes_paired_changes(self):
        neighbors = choose_neighbors(proposals(), SEED, 3)
        result = summarize_neighborhood(
            self.synthetic_trials(neighbors), neighbors, 3,
        )
        self.assertEqual(result[0]["id"], "seed")
        self.assertEqual(len(result), 4)
        self.assertEqual(result[1]["decode_faster_pairs"], 3)
        self.assertEqual(result[1]["paired_differences"][0]["decode_delta_tps"], 0.05)
        self.assertEqual(result[1]["paired_differences"][0]["vram_peak_delta_mib"], 4)

    def test_summary_refuses_incomplete_or_duplicate_or_bad_mapping(self):
        neighbors = choose_neighbors(proposals(), SEED, 3)
        rows = self.synthetic_trials(neighbors)
        with self.assertRaises(ValueError):
            summarize_neighborhood(rows[:-1], neighbors, 3)
        with self.assertRaises(ValueError):
            summarize_neighborhood(rows + [rows[0]], neighbors, 3)
        rows = self.synthetic_trials(neighbors)
        rows[1]["mapping_verified"] = False
        with self.assertRaises(ValueError):
            summarize_neighborhood(rows, neighbors, 3)

    def test_summary_refuses_short_and_nonfinite(self):
        neighbors = choose_neighbors(proposals(), SEED, 3)
        rows = self.synthetic_trials(neighbors)
        rows[1]["generated_tokens"] = 10
        with self.assertRaises(ValueError):
            summarize_neighborhood(rows, neighbors, 3)
        rows = self.synthetic_trials(neighbors)
        rows[1]["decode_tps"] = float("inf")
        with self.assertRaises(ValueError):
            summarize_neighborhood(rows, neighbors, 3)


if __name__ == "__main__":
    unittest.main()
