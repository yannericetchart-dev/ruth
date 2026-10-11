"""Pure/offline tests for all-family, bidirectional GGUF placement proposals."""

import re
import unittest

from calibration.mapping_candidates import propose_mapping_candidates


class MappingCandidateTests(unittest.TestCase):
    @staticmethod
    def layout():
        records = [
            {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100},
            {"name": "output_norm.weight", "group": "output", "block": None, "bytes": 8},
            {"name": "output.weight", "group": "output", "block": None, "bytes": 100},
        ]
        for block in range(4):
            records.extend([
                {"name": f"blk.{block}.attn_q.weight", "group": "attention",
                 "block": block, "bytes": 30},
                {"name": f"blk.{block}.attn_k.weight", "group": "attention",
                 "block": block, "bytes": 20},
                {"name": f"blk.{block}.ffn_down.weight", "group": "feed_forward",
                 "block": block, "bytes": 40},
                {"name": f"blk.{block}.ffn_up.weight", "group": "feed_forward",
                 "block": block, "bytes": 80},
                {"name": f"blk.{block}.attn_norm.weight", "group": "normalization",
                 "block": block, "bytes": 2},
                {"name": f"blk.{block}.ffn_norm.weight", "group": "normalization",
                 "block": block, "bytes": 2},
                {"name": f"blk.{block}.custom.weight", "group": "other_block",
                 "block": block, "bytes": 10},
            ])
        return {"metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
                "tensors": records}

    def test_directions_and_all_families(self):
        report = propose_mapping_candidates(self.layout(), 3)
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_gpu"], [2, 3])
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_cpu"], [0, 1])
        self.assertTrue(report["output_rule"]["require_gpu"])
        self.assertFalse(report["profile_generated"])
        self.assertGreater(report["counts_by_direction"]["gpu_to_host"], 0)
        self.assertGreater(report["counts_by_direction"]["host_to_gpu"], 0)
        self.assertGreater(report["counts_by_direction"]["swap"], 0)
        self.assertEqual(set(report["supported_families"]), {
            "attention", "feed_forward", "normalization", "other_block",
        })
        self.assertEqual(set(report["counts_by_family"]), set(report["supported_families"]))

    def test_exact_regex_and_protected_weights(self):
        report = propose_mapping_candidates(self.layout(), 3)
        inventory = [t["name"] for t in self.layout()["tensors"]]
        for candidate in report["candidates"]:
            self.assertNotIn("token_embd.weight", candidate["tensor_names"])
            self.assertFalse(any(name.startswith("output.") or
                                 name.startswith("output_norm.")
                                 for name in candidate["tensor_names"]))
            for rule in candidate["rules"]:
                actual = [n for n in inventory if re.search(rule["pattern"], n)]
                expected = candidate["host_tensor_names"] if rule["device"] == "cpu" else candidate["gpu_tensor_names"]
                self.assertEqual(sorted(actual), expected)
            if candidate["direction"] == "swap":
                self.assertLessEqual(candidate["estimated_vram_weight_delta_bytes"], 0)

    def test_group_rules_and_host_gpu_promotions(self):
        report = propose_mapping_candidates(self.layout(), 3)
        groups = [c for c in report["candidates"] if c["kind"] == "block_family"]
        self.assertTrue(any(c["family"] == "attention" and
                            c["direction"] == "gpu_to_host" for c in groups))
        self.assertTrue(any(c["family"] == "feed_forward" and
                            c["direction"] == "host_to_gpu" for c in groups))

    def test_output_only_offload_can_promote_but_not_evict(self):
        report = propose_mapping_candidates(self.layout(), 1)
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_gpu"], [])
        self.assertFalse(any(c["direction"] == "gpu_to_host" for c in report["candidates"]))
        self.assertFalse(any(c["direction"] == "swap" for c in report["candidates"]))
        self.assertTrue(any(c["direction"] == "host_to_gpu" for c in report["candidates"]))

    def test_full_offload_cannot_promote(self):
        report = propose_mapping_candidates(self.layout(), 5)
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_gpu"], [0, 1, 2, 3])
        self.assertFalse(any(c["direction"] == "host_to_gpu" for c in report["candidates"]))
        self.assertFalse(any(c["direction"] == "swap" for c in report["candidates"]))

    def test_reject_invalid_layers(self):
        for value in (-1, 0, 6):
            with self.assertRaises(ValueError):
                propose_mapping_candidates(self.layout(), value)

    def test_duplicate_tensor_names_are_rejected(self):
        layout = self.layout()
        layout["tensors"].append(dict(layout["tensors"][0]))
        with self.assertRaises(ValueError):
            propose_mapping_candidates(layout, 2)


if __name__ == "__main__":
    unittest.main()
