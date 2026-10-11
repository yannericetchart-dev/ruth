"""Unit tests for offline summarization of granular CUDA benchmarks."""

import unittest

from calibration.mapping_performance_compare import summarize_trials


class PerformanceCompareTests(unittest.TestCase):
    @staticmethod
    def sample():
        candidates = [
            {"id": "baseline", "direction": "unchanged", "family": "all"},
            {"id": "host", "direction": "gpu_to_host", "family": "attention"},
            {"id": "gpu", "direction": "host_to_gpu", "family": "attention"},
            {"id": "swap", "direction": "swap", "family": "attention"},
        ]
        trials = []
        for ident, speed, memory in [
            ("baseline", 16.5, 4967.3),
            ("host", 16.3, 4963.3),
            ("gpu", 16.7, 4971.3),
            ("swap", 16.4, 4967.3),
        ]:
            for run in (1, 2):
                trials.append({
                    "run": run, "candidate_id": ident, "success": True,
                    "decode_tps": speed, "prefill_tps": 170.0,
                    "vram_peak_mib": memory, "generated_tokens": 128,
                })
        return trials, candidates

    def test_medians_and_deltas_are_measured(self):
        trials, candidates = self.sample()
        reports = summarize_trials(trials, candidates, 2)
        self.assertEqual(len(reports), 4)
        self.assertEqual(reports[0]["median_decode_tps"], 16.5)
        self.assertEqual(reports[0]["decode_change_percent_vs_baseline"], 0)
        self.assertEqual(reports[1]["sampled_vram_peak_change_mib_vs_baseline"], -4.0)
        self.assertGreater(reports[2]["decode_change_percent_vs_baseline"], 0)
        self.assertEqual(reports[3]["sampled_vram_peak_change_mib_vs_baseline"], 0)

    def test_missing_trials_rejected(self):
        trials, candidates = self.sample()
        trials.pop()
        with self.assertRaises(ValueError):
            summarize_trials(trials, candidates, 2)

    def test_failed_trial_rejected(self):
        trials, candidates = self.sample()
        trials[3]["success"] = False
        with self.assertRaises(ValueError):
            summarize_trials(trials, candidates, 2)

    def test_missing_decode_metric_rejected(self):
        trials, candidates = self.sample()
        trials[1]["decode_tps"] = None
        with self.assertRaises(ValueError):
            summarize_trials(trials, candidates, 2)

    def test_too_short_generation_rejected(self):
        trials, candidates = self.sample()
        trials[2]["generated_tokens"] = 12
        with self.assertRaises(ValueError):
            summarize_trials(trials, candidates, 2)


if __name__ == "__main__":
    unittest.main()
