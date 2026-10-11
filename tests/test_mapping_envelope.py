"""Offline tests of the context/offload VRAM envelope search."""

import unittest
from unittest.mock import patch

from calibration.mapping_envelope import envelope, parse_cases, summarize_envelope


def samples(cases, rounds=2):
    rows = []
    for idx, case in enumerate(cases):
        for run in range(1, rounds + 1):
            for variant in ("baseline", "candidate"):
                is_candidate = variant == "candidate"
                rows.append({
                    "case_id": case["id"], "round": run,
                    "variant": variant, "success": True,
                    "mapping_verified": True if is_candidate else None,
                    "requested_tokens": 256, "generated_tokens": 256,
                    "decode_tps": 21.0 + idx + run / 10 + (0.15 if is_candidate else 0),
                    "prefill_tps": 600 + idx * 10 + (5 if is_candidate else 0),
                    "vram_peak_mib": 6000 + idx * 8 + (32 if is_candidate else 0),
                })
    return rows


class MappingEnvelopeTests(unittest.TestCase):
    def test_parse_explicit_cases_without_production_context(self):
        items = parse_cases(["40:768:6", "39:1024:6"])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["probe_n_ctx"], 768)
        self.assertEqual(items[1]["gpu_layers"], 39)
        self.assertEqual(items[0]["prompt_repeat"], 6)

    def test_rejects_duplicate_incomplete_and_oversized_cases(self):
        for values in ([], ["40:768"], ["40:768:6", "40:768:6"],
                       ["40:256:6"], ["0:1024:1"], ["39:1024:17"],
                       ["nonsense"], ["40:768:6"] * 9):
            with self.assertRaises(ValueError):
                parse_cases(values)

    def test_summary_is_paired_and_reports_headroom(self):
        cases = parse_cases(["40:768:6", "39:1024:6"])
        summary = summarize_envelope(samples(cases), cases, 2, 6141)
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary[0]["complete"])
        self.assertEqual(summary[0]["paired_rounds"], 2)
        self.assertAlmostEqual(summary[0]["median_decode_delta_tps"], 0.15)
        self.assertEqual(summary[0]["decode_faster_pairs"], 2)
        self.assertEqual(summary[0]["min_sampled_headroom_mib"], 109)
        self.assertEqual(summary[1]["min_sampled_headroom_mib"], 101)

    def test_bad_mapping_metrics_or_missing_trial_rejected_or_incomplete(self):
        cases = parse_cases(["39:1024:6"])
        data = samples(cases)
        data[1]["mapping_verified"] = False
        with self.assertRaises(ValueError):
            summarize_envelope(data, cases, 2, 6141)
        data = samples(cases)
        data[0]["decode_tps"] = float("nan")
        with self.assertRaises(ValueError):
            summarize_envelope(data, cases, 2, 6141)
        data = samples(cases)[:-1]
        result = summarize_envelope(data, cases, 2, 6141)[0]
        self.assertFalse(result["complete"])
        self.assertEqual(result["paired_rounds"], 1)

    def test_worker_receives_each_diagnostic_context_and_preserves_pairs(self):
        cases = parse_cases(["40:768:6", "39:1024:6"])
        seed = {
            "id": "dual_promotion",
            "component_ids": [
                "single:host_to_gpu:blk.0.ffn_gate.weight",
                "single:host_to_gpu:blk.0.attn_k.weight",
            ],
        }
        status = {
            "ready": True, "errors": [], "foreign_processes": [],
            "memory": {"total_mib": 6141.0},
        }
        requests = []

        def fake_benchmark(**kwargs):
            requests.append(kwargs)
            variant = kwargs.get("combination_ids") is not None
            return {
                "success": True,
                "probe_n_ctx": kwargs["probe_n_ctx"],
                "mapping_verification": {"verified": True} if variant else None,
                "measurements": {
                    "generated_tokens": 256,
                    "prompt_tokens": 315,
                    "decode_tokens_per_second": 22.1 if variant else 22.0,
                    "prefill_tokens_per_second": 620.0,
                    "vram_peak_sampled_mib": 6033.0 if variant else 6001.0,
                },
            }

        with (
            patch("calibration.mapping_envelope.GPUManager.calibration_status",
                  return_value=status),
            patch("calibration.mapping_envelope.ModelManager") as manager,
            patch("calibration.mapping_envelope.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.mapping_envelope.choose_compound_trials",
                  return_value=[seed]),
            patch("calibration.mapping_envelope.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.mapping_envelope.benchmark",
                  side_effect=fake_benchmark),
        ):
            report = envelope(cases, rounds=2)
        self.assertTrue(report["success"])
        self.assertFalse(report["profile_generated"])
        self.assertTrue(report["all_cases_complete"])
        self.assertIsNone(report["selected_n_ctx"])
        self.assertEqual(len(requests), 8)
        self.assertEqual(len(report["trials"]), 8)
        self.assertEqual([x["gpu_layers"] for x in requests], [40,40,39,39,39,39,40,40])
        self.assertEqual(
            [x["probe_n_ctx"] for x in requests],
            [768,768,1024,1024,1024,1024,768,768]
        )
        self.assertEqual(report["summary"][0]["paired_rounds"], 2)
        self.assertEqual(report["summary"][1]["paired_rounds"], 2)
        self.assertEqual(manager.return_value.inspect.call_count, 2)

    def test_native_candidate_oom_is_recorded_and_other_case_runs(self):
        cases = parse_cases(["40:768:6", "39:1024:6"])
        seed = {"id": "dual_promotion", "component_ids": ["a", "b"]}
        status = {
            "ready": True, "errors": [], "foreign_processes": [],
            "memory": {"total_mib": 6141.0},
        }
        requests = []

        def fake_benchmark(**kwargs):
            requests.append(kwargs)
            variant = kwargs.get("combination_ids") is not None
            failing = kwargs["gpu_layers"] == 40 and variant
            if failing:
                return {
                    "success": False, "error": "CUDA allocation failed",
                    "failure_type": "native_signal", "suspected_cuda_oom": True,
                }
            return {
                "success": True, "probe_n_ctx": kwargs["probe_n_ctx"],
                "mapping_verification": {"verified": True} if variant else None,
                "measurements": {
                    "generated_tokens": 256, "prompt_tokens": 315,
                    "decode_tokens_per_second": 22.0,
                    "prefill_tokens_per_second": 600.0,
                    "vram_peak_sampled_mib": 5900.0,
                }
            }

        with (
            patch("calibration.mapping_envelope.GPUManager.calibration_status",
                  return_value=status),
            patch("calibration.mapping_envelope.ModelManager"),
            patch("calibration.mapping_envelope.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.mapping_envelope.choose_compound_trials",
                  return_value=[seed]),
            patch("calibration.mapping_envelope.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.mapping_envelope.benchmark",
                  side_effect=fake_benchmark),
        ):
            report = envelope(cases, rounds=2)

        self.assertTrue(report["measurement_completed"])
        self.assertFalse(report["all_cases_complete"])
        self.assertFalse(report["summary"][0]["complete"])
        self.assertTrue(report["summary"][0]["failures"][0]["suspected_cuda_oom"])
        self.assertTrue(report["summary"][1]["complete"])
        self.assertEqual(len([r for r in requests if r["gpu_layers"] == 40]), 2)
        self.assertEqual(len([r for r in requests if r["gpu_layers"] == 39]), 4)


if __name__ == "__main__":
    unittest.main()
