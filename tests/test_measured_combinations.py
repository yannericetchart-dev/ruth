"""Static tests of the high-offload, compounded candidate experiment."""

import unittest

from calibration.mapping_candidates import propose_mapping_candidates
from calibration.measured_combinations import (
    choose_compound_trials, summarize_compound_trials,
)


class MeasuredCombinationTests(unittest.TestCase):
    @staticmethod
    def fixture():
        tensors = [
            {"name": "token_embd.weight", "group": "embedding", "block": None, "bytes": 100}
        ]
        for block in range(4):
            tensors.extend([
                {"name": f"blk.{block}.ffn_down.weight", "group": "feed_forward", "block": block, "bytes": 20},
                {"name": f"blk.{block}.ffn_gate.weight", "group": "feed_forward", "block": block, "bytes": 30},
                {"name": f"blk.{block}.attn_k.weight", "group": "attention", "block": block, "bytes": 10},
                {"name": f"blk.{block}.attn_q.weight", "group": "attention", "block": block, "bytes": 12},
            ])
        return propose_mapping_candidates({
            "metadata": {"general.architecture": "gemma2", "gemma2.block_count": 4},
            "tensors": tensors,
        }, 3)

    def test_four_nonoverlapping_hypotheses(self):
        definitions = choose_compound_trials(self.fixture())
        self.assertEqual([c["id"] for c in definitions], [
            "baseline", "dual_promotion", "cross_family_exchange",
            "reclaimed_ffn_capacity"
        ])
        self.assertEqual(len(definitions[1]["component_ids"]), 2)
        self.assertEqual(len(definitions[2]["component_ids"]), 2)
        self.assertEqual(len(definitions[3]["component_ids"]), 3)
        self.assertEqual(len(definitions[3]["tensor_names"]), len(set(definitions[3]["tensor_names"])))

    def test_pairwise_deltas_are_descriptive(self):
        definitions = choose_compound_trials(self.fixture())
        trials = []
        for iteration in (1, 2, 3):
            for definition in definitions:
                baseline = definition["id"] == "baseline"
                trials.append({
                    "run": iteration, "trial_id": definition["id"], "success": True,
                    "decode_tps": 20.0 if baseline else 21.0,
                    "prefill_tps": 280.0 if baseline else 276.0,
                    "vram_peak_mib": 5900.0 if baseline else 5920.0,
                })
        summary = summarize_compound_trials(trials, definitions, 3)
        self.assertEqual(len(summary), 4)
        self.assertFalse(summary[0]["decode_faster_in_every_pair"])
        self.assertTrue(all(t["decode_faster_in_every_pair"] for t in summary[1:]))
        self.assertEqual(summary[1]["paired_decode_delta_tps"], [1, 1, 1])
        self.assertEqual(summary[1]["paired_prefill_delta_tps"], [-4, -4, -4])

    def test_missing_pair_rejected(self):
        definitions = choose_compound_trials(self.fixture())
        trials = []
        for iteration in (1, 2):
            for definition in definitions:
                if iteration == 2 and definition["id"] == "dual_promotion":
                    continue
                trials.append({
                    "run": iteration, "trial_id": definition["id"],
                    "success": True, "decode_tps": 20.0,
                    "prefill_tps": 280.0, "vram_peak_mib": 5900.0,
                })
        with self.assertRaises(ValueError):
            summarize_compound_trials(trials, definitions, 2)


if __name__ == "__main__":
    unittest.main()
