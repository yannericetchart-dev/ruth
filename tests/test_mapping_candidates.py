"""Unit tests for offline granular-placement candidate generation."""

import re
import unittest

from calibration.mapping_candidates import propose_mapping_candidates


class MappingCandidateTests(unittest.TestCase):
    @staticmethod
    def layout():
        tensors = [
            {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100},
        ]
        for block in range(4):
            tensors.extend([
                {"name": f"blk.{block}.attn_q.weight", "group": "attention", "block": block, "bytes": 30},
                {"name": f"blk.{block}.ffn_down.weight", "group": "feed_forward", "block": block, "bytes": 40},
                {"name": f"blk.{block}.ffn_up.weight", "group": "feed_forward", "block": block, "bytes": 80},
                {"name": f"blk.{block}.attn_norm.weight", "group": "normalization", "block": block, "bytes": 2},
            ])
        return {"metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
                "tensors": tensors}

    def test_only_last_blocks_and_output_preserved(self):
        result = propose_mapping_candidates(self.layout(), 3)
        self.assertEqual(result["baseline_mapping"]["repeating_blocks_gpu"], [2, 3])
        self.assertEqual(result["baseline_mapping"]["repeating_blocks_cpu"], [0, 1])
        self.assertTrue(result["output_rule"]["require_gpu"])
        self.assertFalse(result["profile_generated"])
        self.assertTrue(all(
            tensor.startswith(("blk.2.", "blk.3."))
            for candidate in result["candidates"]
            for tensor in candidate["tensor_names"]
        ))
        self.assertFalse(any(
            tensor.startswith("token_embd.") or "attn" in tensor
            for candidate in result["candidates"]
            for tensor in candidate["tensor_names"]
        ))

    def test_regexes_match_exact_candidate_tensors(self):
        result = propose_mapping_candidates(self.layout(), 5)
        inventory = [t["name"] for t in self.layout()["tensors"]]
        for candidate in result["candidates"]:
            for rule in candidate["rules"]:
                matched = [name for name in inventory if re.search(rule["pattern"], name)]
                self.assertEqual(sorted(matched), sorted(candidate["tensor_names"]))
        self.assertEqual(result["baseline_mapping"]["repeating_blocks_gpu"], [0, 1, 2, 3])

    def test_offload_one_is_output_only(self):
        result = propose_mapping_candidates(self.layout(), 1)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["baseline_mapping"]["repeating_blocks_gpu"], [])

    def test_out_of_range_rejected(self):
        for layers in (0, -1, 6):
            with self.assertRaises(ValueError):
                propose_mapping_candidates(self.layout(), layers)

    def test_repeated_tensor_names_rejected(self):
        layout = self.layout()
        layout["tensors"].append(layout["tensors"][0].copy())
        with self.assertRaises(ValueError):
            propose_mapping_candidates(layout, 3)


if __name__ == "__main__":
    unittest.main()
