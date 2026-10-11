"""Offline tests for complete GGUF tensor inventory and placement proposals."""

import re
import unittest

from calibration.mapping_candidates import propose_mapping_candidates


class MappingCandidateTests(unittest.TestCase):
    @staticmethod
    def layout():
        records = [
            {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100},
            {"name": "output_norm.weight", "group": "other", "block": None, "bytes": 8},
            {"name": "output.weight", "group": "output", "block": None, "bytes": 100},
            {"name": "extra_scale", "group": "other", "block": None, "bytes": 3},
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
                {"name": f"blk.{block}.custom.scale", "group": "other_block",
                 "block": block, "bytes": 10},
            ])
        return {"metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
                "tensors": records}

    def test_directions_and_complete_inventory(self):
        layout = self.layout()
        report = propose_mapping_candidates(layout, 3)
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_gpu"], [2, 3])
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_cpu"], [0, 1])
        self.assertFalse(report["output_rule"]["require_gpu"])
        self.assertFalse(report["profile_generated"])
        self.assertEqual(report["inventory_tensor_count"], len(layout["tensors"]))
        self.assertEqual(
            set(t["name"] for t in report["tensor_inventory"]),
            set(t["name"] for t in layout["tensors"])
        )
        self.assertIn("embedding", report["supported_families"])
        self.assertIn("output", report["supported_families"])
        self.assertIn("attention", report["supported_families"])
        self.assertIn("feed_forward", report["supported_families"])
        self.assertIn("normalization", report["supported_families"])
        self.assertIn("other_block", report["supported_families"])
        self.assertGreater(report["counts_by_direction"]["gpu_to_host"], 0)
        self.assertGreater(report["counts_by_direction"]["host_to_gpu"], 0)
        self.assertGreater(report["counts_by_direction"]["swap"], 0)

    def test_exact_regex_and_tied_output_is_never_split(self):
        layout = self.layout()
        report = propose_mapping_candidates(layout, 3)
        inventory = [t["name"] for t in layout["tensors"]]
        tied = [c for c in report["candidates"] if c["kind"] == "tied_input_output"]
        self.assertEqual(len(tied), 2)
        self.assertEqual(
            set(c["output_weight_target"] for c in tied), {"host", "gpu"}
        )
        for c in tied:
            self.assertEqual(c["tensor_names"], ["token_embd.weight"])
            self.assertEqual(
                c["expected_native_log_occurrences"], {"token_embd.weight": 2}
            )
            self.assertIsNone(c["estimated_vram_weight_delta_bytes"])

        for candidate in report["candidates"]:
            for rule in candidate["rules"]:
                actual = sorted(n for n in inventory if re.search(rule["pattern"], n))
                expected = (candidate["host_tensor_names"] if rule["device"] == "cpu"
                            else candidate["gpu_tensor_names"])
                self.assertEqual(actual, expected)
            if candidate["direction"] == "swap":
                self.assertLessEqual(candidate["estimated_vram_weight_delta_bytes"], 0)
            if "token_embd.weight" in candidate["tensor_names"]:
                self.assertEqual(candidate["kind"], "tied_input_output")

    def test_output_norm_and_nonweight_tensor_have_candidates(self):
        report = propose_mapping_candidates(self.layout(), 3)
        candidates = report["candidates"]
        self.assertTrue(any(
            "output_norm.weight" in c["host_tensor_names"]
            for c in candidates
        ))
        self.assertTrue(any(
            "blk.2.custom.scale" in c["host_tensor_names"]
            for c in candidates
        ))
        self.assertTrue(any(
            "blk.0.custom.scale" in c["gpu_tensor_names"]
            for c in candidates
        ))
        self.assertTrue(any(
            "extra_scale" in c["host_tensor_names"]
            for c in candidates
        ))
        self.assertTrue(any(
            "extra_scale" in c["gpu_tensor_names"]
            for c in candidates
        ))

    def test_block_family_swaps(self):
        report = propose_mapping_candidates(self.layout(), 3)
        groups = [c for c in report["candidates"] if c["kind"] == "block_family"]
        self.assertTrue(any(c["family"] == "attention" and
                            c["direction"] == "gpu_to_host" for c in groups))
        self.assertTrue(any(c["family"] == "feed_forward" and
                            c["direction"] == "host_to_gpu" for c in groups))

    def test_output_only_offload_promotions_and_tied_choices(self):
        report = propose_mapping_candidates(self.layout(), 1)
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_gpu"], [])
        self.assertFalse(any(
            c["direction"] == "swap" for c in report["candidates"]
        ))
        self.assertTrue(any(
            c["direction"] == "host_to_gpu" for c in report["candidates"]
        ))
        self.assertTrue(any(
            c["direction"] == "coupled_to_host" for c in report["candidates"]
        ))

    def test_full_offload_has_no_repeating_layer_promotions(self):
        report = propose_mapping_candidates(self.layout(), 5)
        self.assertEqual(report["baseline_mapping"]["repeating_blocks_gpu"], [0, 1, 2, 3])
        self.assertFalse(any(
            c["direction"] == "swap" for c in report["candidates"]
        ))
        self.assertTrue(any(c["direction"] == "coupled_to_gpu"
                            for c in report["candidates"]))

    def test_reject_invalid_layers(self):
        for layers in (-1, 0, 6):
            with self.assertRaises(ValueError):
                propose_mapping_candidates(self.layout(), layers)

    def test_duplicate_tensor_names_are_rejected(self):
        layout = self.layout()
        layout["tensors"].append(dict(layout["tensors"][0]))
        with self.assertRaises(ValueError):
            propose_mapping_candidates(layout, 2)


if __name__ == "__main__":
    unittest.main()
