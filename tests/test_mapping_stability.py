"""Unit tests for paired ABBA/BAAB summaries and optional GPU NVML telemetry."""

import unittest

from calibration.gpu_telemetry import gpu_snapshot, summarize_gpu_telemetry
from calibration.mapping_stability import schedule, summarize_stability


class FakeNvml:
    class NVMLError(Exception):
        pass

    NVML_TEMPERATURE_GPU = 0
    NVML_CLOCK_SM = 1

    @staticmethod
    def nvmlDeviceGetTemperature(handle, kind):
        return 65

    @staticmethod
    def nvmlDeviceGetClockInfo(handle, kind):
        return 1600

    @staticmethod
    def nvmlDeviceGetPowerUsage(handle):
        return 82000


def rows(rounds=3):
    data = []
    for batch, order in enumerate(schedule(rounds), start=1):
        for pos, variant in enumerate(order, start=1):
            data.append({
                "round": batch,
                "position": pos,
                "variant": variant,
                "success": True,
                "mapping_verified": True if variant == "candidate" else None,
                "requested_tokens": 256,
                "generated_tokens": 256,
                "decode_tps": 22.0 + (0.2 if variant == "candidate" else 0.0)
                + 0.04 * pos,
                "prefill_tps": 270.0 + (5.0 if variant == "candidate" else 0.0)
                - 0.8 * pos,
                "vram_peak_mib": 5941.3 if variant == "candidate" else 5909.3,
            })
    return data


class StabilityProtocolTests(unittest.TestCase):
    def test_counterbalanced_schedule(self):
        self.assertEqual(schedule(2), [
            ["baseline", "candidate", "candidate", "baseline"],
            ["candidate", "baseline", "baseline", "candidate"],
        ])
        with self.assertRaises(ValueError):
            schedule(1)
        with self.assertRaises(ValueError):
            schedule(9)

    def test_linear_order_drift_cancels(self):
        report = summarize_stability(rows(), 3, 6141)
        self.assertEqual(report["decode_positive_rounds"], 3)
        self.assertAlmostEqual(report["median_decode_delta_tps"], 0.2)
        self.assertAlmostEqual(report["median_prefill_delta_tps"], 5.0)
        self.assertEqual(report["sampled_candidate_headroom_mib"], 199.7)
        self.assertFalse(report["candidate_selected_for_runtime"])

    def test_rejects_missing_or_duplicate_rows(self):
        data = rows()
        with self.assertRaises(ValueError):
            summarize_stability(data[:-1], 3, 6141)
        with self.assertRaises(ValueError):
            summarize_stability(data + [data[0]], 3, 6141)

    def test_rejects_wrong_order_and_unverified_candidate(self):
        data = rows()
        data[0]["variant"] = "candidate"
        with self.assertRaises(ValueError):
            summarize_stability(data, 3, 6141)
        data = rows()
        data[1]["mapping_verified"] = False
        with self.assertRaises(ValueError):
            summarize_stability(data, 3, 6141)

    def test_rejects_incomplete_generation(self):
        data = rows()
        data[1]["generated_tokens"] = 140
        with self.assertRaises(ValueError):
            summarize_stability(data, 3, 6141)

    def test_gpu_telemetry_collects_supported_metrics(self):
        sample = gpu_snapshot(FakeNvml, object())
        self.assertEqual(sample["gpu_temperature_c"], 65)
        self.assertEqual(sample["sm_clock_mhz"], 1600)
        self.assertEqual(sample["power_watts"], 82.0)
        summ = summarize_gpu_telemetry({"decode": [sample, sample]})
        self.assertEqual(summ["decode"]["sm_clock_mhz"]["median"], 1600)
        self.assertEqual(summ["decode"]["sample_count"], 2)

    def test_missing_gpu_telemetry_is_null_not_failure(self):
        class Unsupported(FakeNvml):
            @staticmethod
            def nvmlDeviceGetClockInfo(handle, kind):
                raise FakeNvml.NVMLError("unsupported")
        sample = gpu_snapshot(Unsupported, object())
        self.assertIsNone(sample["sm_clock_mhz"])
        summ = summarize_gpu_telemetry({"decode": [sample]})
        self.assertIsNone(summ["decode"]["sm_clock_mhz"])


if __name__ == "__main__":
    unittest.main()
