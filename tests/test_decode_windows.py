"""CPU-only tests for decode-window timing and context occupancy diagnostics."""

import unittest

from calibration.decode_windows import summarize_decode_windows


class DecodeWindowTests(unittest.TestCase):
    def test_constant_rate_equal_windows(self):
        stamps = [i * 0.1 for i in range(9)]
        data = summarize_decode_windows(stamps, prompt_tokens=600, n_ctx=1024)
        self.assertEqual(data["window_count"], 4)
        self.assertEqual(data["decode_steps_total"], 8)
        self.assertEqual(data["generated_tokens"], 9)
        self.assertEqual([w["decode_steps"] for w in data["windows"]], [2, 2, 2, 2])
        self.assertTrue(all(w["decode_tps"] == 10 for w in data["windows"]))
        self.assertEqual(data["windows"][-1]["ending_context_tokens"], 608)
        self.assertEqual(data["last_minus_first_tps"], 0)

    def test_later_decode_slows_even_with_same_prompt(self):
        stamps = [0.0]
        for elapsed in [0.05] * 4 + [0.1] * 4:
            stamps.append(stamps[-1] + elapsed)
        data = summarize_decode_windows(stamps, prompt_tokens=600, n_ctx=1024)
        self.assertEqual(data["first_window_tps"], 20)
        self.assertEqual(data["last_window_tps"], 10)
        self.assertEqual(data["last_minus_first_tps"], -10)

    def test_unequal_division_tracks_all_intervals(self):
        stamps = [i * 0.25 for i in range(12)]
        data = summarize_decode_windows(stamps, 512, 1024)
        self.assertEqual(sum(w["decode_steps"] for w in data["windows"]), 11)
        self.assertEqual(data["end_context_fraction"], round(523 / 1024, 4))

    def test_rejects_nonmonotone_nan_or_insufficient(self):
        with self.assertRaises(ValueError):
            summarize_decode_windows([0, 0, 0.2, 0.3, 0.4], 315, 768)
        with self.assertRaises(ValueError):
            summarize_decode_windows([0, 0.1, float("nan"), 0.3, 0.4], 315, 768)
        with self.assertRaises(ValueError):
            summarize_decode_windows([0, 0.1, 0.2], 315, 768)
        with self.assertRaises(ValueError):
            summarize_decode_windows([i * 0.1 for i in range(9)], 1024, 1024)
        with self.assertRaises(ValueError):
            summarize_decode_windows([i * 0.1 for i in range(9)], 315, 768, 1)


if __name__ == "__main__":
    unittest.main()
