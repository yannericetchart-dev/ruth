"""Offline checks: paired mapping confirmation rejects invalid evidence."""

import unittest

from calibration.mapping_confirmation import summarize_confirmation


def fixture(rounds=3, gain=0.2, prefill=7.0):
    rows = []
    for run in range(1, rounds + 1):
        rows.extend([
            {
                "run": run, "variant": "baseline", "success": True,
                "native_mapping_verified": None,
                "decode_tps": 22.2 + run / 100,
                "prefill_tps": 280.0 + run,
                "vram_peak_mib": 5910,
                "generated_tokens": 128,
            },
            {
                "run": run, "variant": "candidate", "success": True,
                "native_mapping_verified": True,
                "decode_tps": 22.2 + run / 100 + gain,
                "prefill_tps": 280.0 + run + prefill,
                "vram_peak_mib": 5942,
                "generated_tokens": 128,
            },
        ])
    return rows


class MappingConfirmationTests(unittest.TestCase):
    def test_paired_differences_and_memory_are_descriptive(self):
        s = summarize_confirmation(fixture(), 3)
        self.assertEqual(s["decode_improved_pairs"], 3)
        self.assertEqual(s["prefill_improved_pairs"], 3)
        self.assertTrue(s["decode_improved_in_every_pair"])
        self.assertEqual(s["paired_decode_delta_median_tps"], 0.2)
        self.assertTrue(s["descriptive_only"])
        self.assertEqual(s["candidate"]["max_sampled_vram_peak_mib"], 5942)
        self.assertEqual(len(s["paired_observations"]), 3)
        self.assertEqual(s["paired_observations"][1]["vram_peak_delta_mib"], 32)

    def test_negative_gain_does_not_pass_consistency_flag(self):
        s = summarize_confirmation(fixture(gain=-0.1), 3)
        self.assertFalse(s["decode_improved_in_every_pair"])
        self.assertEqual(s["decode_improved_pairs"], 0)

    def test_missing_and_duplicate_pairs_are_rejected(self):
        rows = fixture()
        with self.assertRaises(ValueError):
            summarize_confirmation(rows[:-1], 3)
        with self.assertRaises(ValueError):
            summarize_confirmation(rows + rows[:1], 3)

    def test_unverified_native_candidate_is_rejected(self):
        rows = fixture()
        rows[1]["native_mapping_verified"] = False
        with self.assertRaises(ValueError):
            summarize_confirmation(rows, 3)

    def test_non_finite_or_short_samples_are_rejected(self):
        rows = fixture()
        rows[1]["decode_tps"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_confirmation(rows, 3)
        rows = fixture()
        rows[1]["generated_tokens"] = 12
        with self.assertRaises(ValueError):
            summarize_confirmation(rows, 3)


if __name__ == "__main__":
    unittest.main()
