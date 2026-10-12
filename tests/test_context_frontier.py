"""CPU-only tests for CUDA context/offload adaptive frontier semantics."""

import unittest
from unittest.mock import patch

from calibration.context_frontier import (
    classify_trial, explore_frontier, parse_contexts, summarize_frontier,
)


SEED = {
    "id": "dual_promotion",
    "component_ids": [
        "single:host_to_gpu:blk.0.ffn_gate.weight",
        "single:host_to_gpu:blk.0.attn_k.weight",
    ],
}
STATUS = {
    "ready": True,
    "errors": [],
    "foreign_processes": [],
    "memory": {"total_mib": 6141.0},
}


def fake_outcome(kwargs, fail=None):
    if fail is not None:
        return {
            "success": False,
            "failure_type": "native_signal",
            "suspected_cuda_oom": fail == "oom",
            "error": "CUDA out of memory" if fail == "oom" else "unverified backend failure",
        }
    mapping = kwargs.get("combination_ids") is not None
    prompt_tokens = 1233 if kwargs["probe_n_ctx"] == 1536 else 1743
    fractions = {
        1536: 0.9688,
        2048: 0.9756,
    }
    return {
        "success": True,
        "probe_n_ctx": kwargs["probe_n_ctx"],
        "mapping_verification": {"verified": True} if mapping else None,
        "measurements": {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": 256,
            "decode_tokens_per_second": 20.5 if mapping else 20.3,
            "prefill_tokens_per_second": 600.0,
            "vram_peak_sampled_mib": 6050.0 if mapping else 6018.0,
            "decode_windows": {
                "windows": [{
                    "window": n + 1,
                    "decode_tps": 20.5 + n * 0.01 + (0.2 if mapping else 0),
                    "decode_steps": 64,
                } for n in range(4)],
                "end_context_fraction": fractions[kwargs["probe_n_ctx"]],
            },
        },
    }


class ContextFrontierTests(unittest.TestCase):
    def test_parse_contexts_and_reject_invalid_repetitions(self):
        items = parse_contexts(["1536:24", "2048:34"])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["prompt_repeat"], 24)
        self.assertEqual(items[1]["probe_n_ctx"], 2048)
        for wrong in (
            [], ["1536"], ["1536:-1"], ["512:49"], ["2048:0"],
            ["1536:24", "1536:34"], ["2048:34"] * 6,
        ):
            with self.assertRaises(ValueError):
                parse_contexts(wrong)

    def test_classify_native_verified_and_oom_separately(self):
        kwargs = {"probe_n_ctx": 1536, "combination_ids": ["a", "b"]}
        valid = fake_outcome(kwargs)
        self.assertEqual(classify_trial(valid, True, 1536, 256), (True, None))
        self.assertEqual(
            classify_trial(fake_outcome(kwargs, "oom"), True, 1536, 256),
            (False, "suspected_cuda_oom"),
        )
        self.assertEqual(
            classify_trial(fake_outcome(kwargs, "crash"), True, 1536, 256),
            (False, "worker_failure"),
        )
        valid["mapping_verification"]["verified"] = False
        self.assertEqual(
            classify_trial(valid, True, 1536, 256),
            (False, "native_mapping_mismatch"),
        )

    def test_incomplete_generation_is_not_an_oom(self):
        kwargs = {"probe_n_ctx": 1536, "combination_ids": None}
        out = fake_outcome(kwargs)
        out["measurements"]["generated_tokens"] = 100
        self.assertEqual(
            classify_trial(out, False, 1536, 256),
            (False, "incomplete_diagnostic_measurement"),
        )

    def test_adaptive_retry_uses_native_oom_and_measures_full_batches(self):
        contexts = parse_contexts(["1536:24", "2048:34"])
        calls = []

        def fake_benchmark(**kwargs):
            calls.append(kwargs)
            ctx = kwargs["probe_n_ctx"]
            layers = kwargs["gpu_layers"]
            candidate = kwargs.get("combination_ids") is not None
            if ctx == 1536 and layers == 39 and candidate:
                return fake_outcome(kwargs, "oom")
            if ctx == 2048 and layers == 39:
                return fake_outcome(kwargs, "oom")
            if ctx == 2048 and layers == 38 and candidate:
                return fake_outcome(kwargs, "oom")
            return fake_outcome(kwargs)

        with (
            patch("calibration.context_frontier.GPUManager.calibration_status",
                  return_value=STATUS),
            patch("calibration.context_frontier.ModelManager") as manager,
            patch("calibration.context_frontier.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.context_frontier.choose_compound_trials",
                  return_value=[SEED]),
            patch("calibration.context_frontier.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.context_frontier.benchmark",
                  side_effect=fake_benchmark),
        ):
            result = explore_frontier(
                contexts, initial_gpu_layers=39, minimum_gpu_layers=37, rounds=2
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["measurement_completed"])
        self.assertTrue(result["summary"]["all_contexts_measured_feasible"])
        self.assertFalse(result["profile_generated"])
        self.assertIsNone(result["selected_n_ctx"])
        summaries = result["summary"]["contexts"]
        self.assertEqual(
            [s["highest_tested_feasible_gpu_layers"] for s in summaries], [38, 37]
        )
        self.assertEqual(len(calls), 21)
        self.assertEqual(manager.return_value.inspect.call_count, 5)
        for context in result["contexts"]:
            successful = next(a for a in context["attempts"] if a["complete"])
            self.assertEqual(len(successful["trials"]), 8)
            comparison = successful["comparison"]
            self.assertEqual(comparison["decode_positive_rounds"], 2)
            self.assertEqual(
                comparison["counterbalanced_decode_windows"]["last_quartile_positive_rounds"],
                2,
            )
            self.assertEqual(comparison["sampled_candidate_headroom_mib"], 91)
        self.assertEqual(
            [a["failure_class"] for a in result["contexts"][1]["attempts"]],
            ["suspected_cuda_oom", "suspected_cuda_oom", None],
        )

    def test_unrelated_failure_aborts_instead_of_reduce_offload(self):
        contexts = parse_contexts(["1536:24"])
        calls = []

        def fail(**kwargs):
            calls.append(kwargs)
            return fake_outcome(kwargs, "crash")

        with (
            patch("calibration.context_frontier.GPUManager.calibration_status",
                  return_value=STATUS),
            patch("calibration.context_frontier.ModelManager"),
            patch("calibration.context_frontier.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.context_frontier.choose_compound_trials",
                  return_value=[SEED]),
            patch("calibration.context_frontier.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.context_frontier.benchmark", side_effect=fail),
        ):
            result = explore_frontier(contexts, initial_gpu_layers=39,
                                      minimum_gpu_layers=37, rounds=2)
        self.assertFalse(result["success"])
        self.assertEqual(len(calls), 1)
        self.assertIn("worker_failure", result["error"])
        self.assertEqual(len(result["contexts"][0]["attempts"]), 1)

    def test_all_levels_oom_are_reported_not_mislabelled_success(self):
        contexts = parse_contexts(["1536:24"])
        with (
            patch("calibration.context_frontier.GPUManager.calibration_status",
                  return_value=STATUS),
            patch("calibration.context_frontier.ModelManager"),
            patch("calibration.context_frontier.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.context_frontier.choose_compound_trials",
                  return_value=[SEED]),
            patch("calibration.context_frontier.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.context_frontier.benchmark",
                  side_effect=lambda **kw: fake_outcome(kw, "oom")),
        ):
            result = explore_frontier(contexts, initial_gpu_layers=39,
                                      minimum_gpu_layers=38, rounds=2)
        self.assertTrue(result["measurement_completed"])
        self.assertFalse(result["summary"]["all_contexts_measured_feasible"])
        self.assertIsNone(
            result["summary"]["contexts"][0]["highest_tested_feasible_gpu_layers"]
        )
        self.assertEqual(len(result["contexts"][0]["attempts"]), 2)

    @staticmethod
    def ambiguous_context_failure():
        return {
            "success": False,
            "error": "ValueError: Failed to create llama_context",
            "worker_returncode": 0,
            "failure_hint": "ambiguous_context_initialization",
            "suspected_cuda_oom": False,
            "cuda_diagnostics": ["llama_context: failed to initialize"],
            "stderr_tail": "llama_context: failed to initialize",
        }

    def test_ambiguous_context_error_is_not_labelled_oom(self):
        issue = self.ambiguous_context_failure()
        self.assertEqual(
            classify_trial(issue, False, 2048, 256),
            (False, "ambiguous_context_initialization"),
        )
        self.assertFalse(issue["suspected_cuda_oom"])

    def test_ambiguous_context_stops_by_default_but_retains_partial_summary(self):
        contexts = parse_contexts(["2048:34"])
        calls = []

        def fake_benchmark(**kwargs):
            calls.append(kwargs)
            return self.ambiguous_context_failure()

        with (
            patch("calibration.context_frontier.GPUManager.calibration_status",
                  return_value=STATUS),
            patch("calibration.context_frontier.ModelManager"),
            patch("calibration.context_frontier.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.context_frontier.choose_compound_trials",
                  return_value=[SEED]),
            patch("calibration.context_frontier.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.context_frontier.benchmark",
                  side_effect=fake_benchmark),
        ):
            result = explore_frontier(contexts, initial_gpu_layers=39,
                                      minimum_gpu_layers=37, rounds=2)
        self.assertFalse(result["success"])
        self.assertFalse(result["retry_ambiguous_context"])
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(result["summary"])
        self.assertEqual(result["contexts"][0]["attempts"][0]["failure_class"],
                         "ambiguous_context_initialization")
        trial = result["contexts"][0]["attempts"][0]["trials"][0]
        self.assertIn("llama_context", trial["stderr_tail"])
        self.assertFalse(trial["suspected_cuda_oom"])

    def test_explicit_opt_in_retries_ambiguous_context_without_reclassifying_it(self):
        contexts = parse_contexts(["2048:34"])
        calls = []

        def fake_benchmark(**kwargs):
            calls.append(kwargs)
            if kwargs["gpu_layers"] == 39:
                return self.ambiguous_context_failure()
            return fake_outcome(kwargs)

        with (
            patch("calibration.context_frontier.GPUManager.calibration_status",
                  return_value=STATUS),
            patch("calibration.context_frontier.ModelManager"),
            patch("calibration.context_frontier.propose_mapping_candidates",
                  return_value={"candidates": []}),
            patch("calibration.context_frontier.choose_compound_trials",
                  return_value=[SEED]),
            patch("calibration.context_frontier.compose_candidates",
                  return_value={"rules": []}),
            patch("calibration.context_frontier.benchmark",
                  side_effect=fake_benchmark),
        ):
            result = explore_frontier(
                contexts, initial_gpu_layers=39,
                minimum_gpu_layers=37, rounds=2,
                retry_ambiguous_context=True,
            )
        self.assertTrue(result["success"])
        self.assertTrue(result["retry_ambiguous_context"])
        self.assertEqual(len(calls), 9)
        self.assertEqual(
            result["summary"]["contexts"][0]["highest_tested_feasible_gpu_layers"],
            38,
        )
        self.assertEqual(result["contexts"][0]["attempts"][0]["failure_class"],
                         "ambiguous_context_initialization")
        self.assertEqual(result["contexts"][0]["attempts"][1]["failure_class"], None)
        self.assertTrue(result["contexts"][0]["attempts"][1]["complete"])
        self.assertFalse(result["profile_generated"])
        self.assertIsNone(result["selected_n_ctx"])

    def test_summary_disallows_duplicate_successful_offload_per_context(self):
        duplicated = [{
            "probe_n_ctx": 1536, "prompt_repeat": 24,
            "attempts": [
                {"gpu_layers": 39, "complete": True, "comparison": {},
                 "failure_class": None, "trials": []},
                {"gpu_layers": 38, "complete": True, "comparison": {},
                 "failure_class": None, "trials": []},
            ],
        }]
        with self.assertRaises(ValueError):
            summarize_frontier(duplicated, 6141)


if __name__ == "__main__":
    unittest.main()
