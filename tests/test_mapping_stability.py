"""Unit tests for paired ABBA/BAAB summaries and optional GPU NVML telemetry."""

import unittest
from unittest.mock import patch

from calibration.benchmark import probe_prompt, validate_probe
from calibration.mapping_stability import run_stability
from calibration.gpu_telemetry import gpu_snapshot, summarize_gpu_telemetry
from calibration.mapping_stability import schedule, summarize_stability
from calibration.window_stability import compare_decode_windows


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

    def test_diagnostic_prompt_preserves_chat_template_and_repeats_body(self):
        once = probe_prompt(1)
        repeated = probe_prompt(6)
        self.assertEqual(once.count("<start_of_turn>user"), 1)
        self.assertEqual(once.count("<start_of_turn>model"), 1)
        self.assertEqual(repeated.count("<start_of_turn>user"), 1)
        self.assertEqual(repeated.count("Spiega dettagliatamente"), 6)
        self.assertGreater(len(repeated), len(once))
        self.assertIn("<start_of_turn>user\n", once)
        self.assertIn("<end_of_turn>\n<start_of_turn>model\n", once)

    def test_diagnostic_context_and_prompt_bounds(self):
        validate_probe(768, 6, 256)
        validate_probe(2048, 34, 256)
        for params in ((256, 1, 128), (8193, 1, 128),
                       (768, 0, 256), (768, 49, 256),
                       (512, 1, 512)):
            with self.assertRaises(ValueError):
                validate_probe(*params)

    def test_probe_parameters_reach_every_worker(self):
        seed = {
            "id": "dual_promotion",
            "component_ids": [
                "single:host_to_gpu:blk.0.ffn_gate.weight",
                "single:host_to_gpu:blk.0.attn_k.weight",
            ],
        }
        settings = {"ready": True, "memory": {"total_mib": 6141.0},
                    "errors": [], "foreign_processes": []}
        calls = []

        def fake_benchmark(**kwargs):
            calls.append(kwargs)
            is_candidate = kwargs.get("combination_ids") is not None
            return {
                "success": True, "probe_n_ctx": kwargs["probe_n_ctx"],
                "mapping_verification": {"verified": True} if is_candidate else None,
                "measurements": {
                    "prompt_tokens": 330,
                    "generated_tokens": 256,
                    "decode_windows": {
                        "windows": [
                            {
                                "window": i + 1,
                                "decode_tps": 22.2 + i * 0.1 + (0.2 if is_candidate else 0.0),
                                "decode_steps": 64,
                            }
                            for i in range(4)
                        ],
                        "end_context_fraction": 0.762,
                    },
                    "decode_tokens_per_second": 22.2 if is_candidate else 22.0,
                    "prefill_tokens_per_second": 280.0,
                    "vram_peak_sampled_mib": 6035.0 if is_candidate else 6003.0,
                },
                "gpu_telemetry_by_phase": {"decode": {}},
            }

        with (
            patch("calibration.mapping_stability.GPUManager.calibration_status",
                  return_value=settings),
            patch("calibration.mapping_stability.ModelManager") as manager,
            patch("calibration.mapping_stability.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.mapping_stability.choose_compound_trials",
                  return_value=[seed]),
            patch("calibration.mapping_stability.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.mapping_stability.benchmark",
                  side_effect=fake_benchmark),
        ):
            report = run_stability(
                gpu_layers=40, rounds=2, tokens=256,
                probe_n_ctx=768, prompt_repeat=6
            )
        self.assertTrue(report["success"])
        self.assertEqual(report["probe_n_ctx"], 768)
        self.assertEqual(report["prompt_repeat"], 6)
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c["probe_n_ctx"] == 768 for c in calls))
        self.assertTrue(all(c["prompt_repeat"] == 6 for c in calls))
        self.assertTrue(all(t["prompt_tokens"] == 330 for t in report["trials"]))
        window_report = report["summary"]["counterbalanced_decode_windows"]
        self.assertEqual(window_report["last_quartile_positive_rounds"], 2)
        self.assertAlmostEqual(window_report["last_quartile_delta_median_tps"], 0.2)
        manager.return_value.inspect.assert_called_once_with(
            "gemma-2-9b-it", n_ctx=768
        )

    def test_counterbalanced_window_deltas_cancel_linear_drift(self):
        data = rows(rounds=2)
        for row in data:
            is_candidate = row["variant"] == "candidate"
            row["prompt_tokens"] = 720
            row["final_context_token_fraction"] = 0.9521
            row["decode_window_details"] = [
                {
                    "window": i + 1,
                    "decode_steps": 64,
                    "decode_tps": 22.0 + 0.1 * row["position"]
                    - 0.3 * i + (0.2 if is_candidate else 0.0),
                }
                for i in range(4)
            ]
        report = summarize_stability(data, 2, 6141)
        result = report["counterbalanced_decode_windows"]
        self.assertEqual(result["prompt_tokens"], 720)
        self.assertEqual(result["last_quartile_positive_rounds"], 2)
        self.assertAlmostEqual(result["last_quartile_delta_median_tps"], 0.2)
        self.assertEqual(
            result["median_candidate_minus_baseline_by_quartile_tps"],
            [0.2, 0.2, 0.2, 0.2],
        )
        self.assertAlmostEqual(
            result["rounds"][0]["baseline_last_minus_first_tps"], -0.9
        )

    def test_missing_or_inconsistent_windows_are_rejected(self):
        data = rows(rounds=2)
        self.assertIsNone(compare_decode_windows(data, 2))
        data[0]["decode_window_details"] = []
        with self.assertRaises(ValueError):
            compare_decode_windows(data, 2)

        data = rows(rounds=2)
        for row in data:
            row["prompt_tokens"] = 720
            row["final_context_token_fraction"] = 0.95
            row["decode_window_details"] = [
                {"window": i + 1, "decode_steps": 64, "decode_tps": 20.0}
                for i in range(4)
            ]
        data[1]["prompt_tokens"] = 719
        with self.assertRaises(ValueError):
            compare_decode_windows(data, 2)
        data[1]["prompt_tokens"] = 720
        data[3]["decode_window_details"][0]["decode_tps"] = -1
        with self.assertRaises(ValueError):
            compare_decode_windows(data, 2)

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
