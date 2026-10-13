"""Planner consumes native measured tensor effects, not GGUF size proxies."""
import json
import unittest

from calibration.kv_offload_candidate_plan import plan_from_report


def example_report():
    original = {
        "prefill_tok_s": 460., "decode_tok_s": 26.,
        "vram_peak_mib": 6100., "vram_total_mib": 6141.,
    }
    a = "blk.0.ffn_down.weight"
    b = "blk.1.ffn_down.weight"
    c = "blk.2.ffn_down.weight"
    rows = [
        ([], 6100., 460., 26.),
        ([a], 6076., 440., 25.),
        ([b], 6078., 445., 25.),
        ([c], 6082., 440., 25.),
        ([a, b], 6050., 425., 24.),
        ([a, b, c], 6026., 400., 23.),
    ]
    measurements = {}
    for i, (names, peak, pre, dec) in enumerate(rows):
        measurements[f"m{i}"] = {
            "n_ctx": 256, "success": True,
            "host_tensor_names": sorted(names),
            "metrics": {
                "vram_peak_mib": peak, "prefill_tok_s": pre,
                "decode_tok_s": dec, "vram_total_mib": 6141.,
            },
        }
    names = sorted([a, b, c])
    validated = {
        "success": True, "n_ctx": 512,
        "host_tensor_names": names, "repetitions": 3,
        "metrics": {
            "prefill_tok_s": 415., "decode_tok_s": 22.5,
            "vram_peak_mib": 6112., "vram_total_mib": 6141.,
        },
        "samples": [{
            "native_placement_verified": True, "gpu_kv_verified": True
        } for _ in range(3)],
    }
    return {
        "schema": "ruth-phase2-adaptive-tensor-greedy-v3",
        "completed": True,
        "parameters": {
            "model_id": "gemma-2-9b-it",
            "model_path": "/tmp/model.gguf",
        },
        "stages": [{
            "from_n_ctx": 256, "successful": True,
            "measured_target_n_ctx": 512,
            "confirmed_host_tensor_names": names,
            "validation": validated,
        }],
        "measured_configs": measurements,
    }


class OfflinePlannerTests(unittest.TestCase):
    def test_plan_is_ranked_and_has_no_minimum_vram_margin(self):
        report = example_report()
        plan = plan_from_report(
            report, source_sha256="fake-sha", min_tensors=2,
            max_tensors=3, shortlist=8,
        )
        self.assertEqual(plan["schema"],
                         "ruth-phase2-offload-candidate-plan-v1")
        self.assertEqual(plan["anchor_n_ctx"], 256)
        self.assertEqual(plan["target_n_ctx"], 512)
        self.assertEqual(
            plan["planning_parameters"]["single_tensor_observations"], 3
        )
        self.assertEqual(
            plan["planning_parameters"]["pair_observations"], 1
        )
        self.assertEqual(
            plan["planning_parameters"]["triple_observations"], 1
        )
        self.assertEqual(
            plan["planning_parameters"]["total_combinations_considered"], 4
        )
        self.assertEqual(len(plan["candidates"]), 4)
        self.assertEqual(
            [c["rank"] for c in plan["candidates"]], [1, 2, 3, 4]
        )
        self.assertTrue(all(
            c["estimation_basis"] in (
                "direct_anchor_observation_plus_seed_ctx_transfer",
                "singleton_plus_empirical_interaction_medians",
            ) for c in plan["candidates"]
        ))
        self.assertTrue(any(
            c["observed_at_anchor"] for c in plan["candidates"]
        ))
        self.assertEqual(plan["source_greedy_sha256"], "fake-sha")

    def test_requires_confirmed_gpu_kv_seed(self):
        report = example_report()
        report["stages"][0]["validation"]["samples"][0][
            "gpu_kv_verified"
        ] = False
        with self.assertRaisesRegex(ValueError, "Seed non confermata"):
            plan_from_report(report, source_sha256="fake")

    def test_rejects_unconfirmed_greedy_run(self):
        report = example_report()
        report["completed"] = False
        with self.assertRaises(ValueError):
            plan_from_report(report, source_sha256="fake")


if __name__ == "__main__":
    unittest.main()
