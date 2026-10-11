"""Pure tests for measured Pareto and diverse candidate screening."""

import unittest

from calibration.mapping_candidates import propose_mapping_candidates
from calibration.mapping_screen import choose_exploration_cohort, measured_pareto


class MappingScreenTests(unittest.TestCase):
    @staticmethod
    def fixture():
        tensors = [{"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100}]
        for block in range(4):
            tensors.extend([
                {"name": f"blk.{block}.attn_k.weight", "group": "attention",
                 "block": block, "bytes": 5},
                {"name": f"blk.{block}.attn_q.weight", "group": "attention",
                 "block": block, "bytes": 12},
                {"name": f"blk.{block}.ffn_up.weight", "group": "feed_forward",
                 "block": block, "bytes": 40},
                {"name": f"blk.{block}.ffn_down.weight", "group": "feed_forward",
                 "block": block, "bytes": 20},
                {"name": f"blk.{block}.attn_norm.weight", "group": "normalization",
                 "block": block, "bytes": 2},
                {"name": f"blk.{block}.ffn_norm.weight", "group": "normalization",
                 "block": block, "bytes": 2},
            ])
        return {"metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
                "tensors": tensors}

    def test_cohort_diverse_and_bounded(self):
        proposals = propose_mapping_candidates(self.fixture(), 3)
        chosen = choose_exploration_cohort(proposals, 6)
        self.assertEqual(len(chosen), 7)
        self.assertEqual(chosen[0]["id"], "baseline")
        self.assertEqual(len({c["id"] for c in chosen}), len(chosen))
        self.assertEqual(set(c["family"] for c in chosen[1:]),
                         {"attention", "feed_forward"})
        self.assertEqual(
            set(c["direction"] for c in chosen[1:]),
            {"gpu_to_host", "host_to_gpu", "swap"},
        )
        self.assertTrue(any(c["kind"] == "block_family" for c in chosen))

    def test_unsupported_offload_one_no_swaps(self):
        proposals = propose_mapping_candidates(self.fixture(), 1)
        chosen = choose_exploration_cohort(proposals, 6)
        self.assertGreater(len(chosen), 1)
        self.assertTrue(all(c["direction"] in ("unchanged", "host_to_gpu")
                            for c in chosen))

    def test_output_only_discovers_promotions_with_no_gpu_evictions(self):
        proposals = propose_mapping_candidates(self.fixture(), 1)
        cohort = choose_exploration_cohort(proposals, 6)
        self.assertGreater(len(cohort), 1)
        self.assertTrue(all(
            c["kind"] != "tied_input_output" and
            c["direction"] in ("unchanged", "host_to_gpu")
            for c in cohort
        ))
        self.assertTrue(any(
            c["family"] == "attention" for c in cohort[1:]
        ))
        self.assertTrue(any(
            c["family"] == "feed_forward" for c in cohort[1:]
        ))

    def test_reject_empty_budget(self):
        proposals = propose_mapping_candidates(self.fixture(), 3)
        with self.assertRaises(ValueError):
            choose_exploration_cohort(proposals, 0)

    def test_measured_pareto_excludes_dominated(self):
        summary = [
            {"candidate_id": "baseline", "median_decode_tps": 16.55,
             "median_prefill_tps": 174.2, "max_sampled_vram_peak_mib": 4967.3},
            {"candidate_id": "host", "median_decode_tps": 16.56,
             "median_prefill_tps": 172.75, "max_sampled_vram_peak_mib": 4963.3},
            {"candidate_id": "gpu", "median_decode_tps": 16.54,
             "median_prefill_tps": 173.30, "max_sampled_vram_peak_mib": 4971.3},
            {"candidate_id": "swap", "median_decode_tps": 16.49,
             "median_prefill_tps": 172.97, "max_sampled_vram_peak_mib": 4967.3},
        ]
        self.assertEqual(measured_pareto(summary), ["baseline", "host"])

    def test_ties_both_remain(self):
        duplicate = {"median_decode_tps": 16, "median_prefill_tps": 170,
                     "max_sampled_vram_peak_mib": 5000}
        summary = [dict(duplicate, candidate_id="a"), dict(duplicate, candidate_id="b")]
        self.assertEqual(measured_pareto(summary), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
