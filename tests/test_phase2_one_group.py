"""CPU-only regression for one-candidate Phase 2 search and comparison."""
import unittest
from pathlib import Path
from unittest.mock import patch
from calibration.phase2_one_group import run_one


class OneGroupPhase2Tests(unittest.TestCase):
    def test_search_then_compare_only_at_extended_context(self):
        metadata = {
            "general.architecture": "gemma2",
            "gemma2.context_length": 520,
        }
        group = {
            "id": "group:attn_norm.weight", "eligible_for_trial": True,
            "gguf_unique_weight_mib": 2.0,
        }
        calls = []

        def probe(path, model_id, trial, *, n_ctx, tokens, timeout):
            calls.append((trial["id"], n_ctx))
            valid = trial["id"] == "baseline_host_kv" or n_ctx <= 400
            return {
                "success": valid,
                "failure_class": None if valid else "native_cuda_oom",
                "measurements": {
                    "prefill_tokens_per_s": 100.0 if trial["id"] != "baseline_host_kv" else 80.0,
                    "decode_tokens_per_s": 24.0 if trial["id"] != "baseline_host_kv" else 20.0,
                    "vram_peak_sampled_mib": 6000.0,
                },
            }

        with (patch("calibration.phase2_one_group.inspect_gguf",
                    return_value={"metadata": metadata}),
              patch("calibration.phase2_one_group.group_inventory",
                    return_value=[group]),
              patch("calibration.phase2_one_group.native_group_trial",
                    side_effect=lambda layout, selected: {
                        "id": selected["id"] if selected else "baseline",
                        "host_tensor_count": 42 if selected else 0,
                    }),
              patch("calibration.phase2_one_group.probe_native_mapping",
                    side_effect=probe)):
            result = run_one(Path("/fake.gguf"), reference_ctx=257)
        self.assertTrue(result["success"])
        self.assertEqual(result["extended_n_ctx"], 400)
        self.assertEqual(result["comparison"]["prompt_tokens"], 272)
        self.assertEqual(result["comparison"]["decode_change_percent"], 20.0)
        self.assertEqual(
            [ctx for mode, ctx in calls if mode == "baseline_host_kv"],
            [400],
        )

    def test_no_extended_context_skips_baseline(self):
        group = {"id": "group:a", "eligible_for_trial": True,
                 "gguf_unique_weight_mib": 1.0}
        calls = []
        def probe(path, model_id, trial, *, n_ctx, tokens, timeout):
            calls.append((trial["id"], n_ctx))
            return {"success": n_ctx == 257,
                    "failure_class": "native_cuda_oom"}
        with (patch("calibration.phase2_one_group.inspect_gguf",
                    return_value={"metadata": {"general.architecture": "gemma2",
                                               "gemma2.context_length": 258}}),
              patch("calibration.phase2_one_group.group_inventory",
                    return_value=[group]),
              patch("calibration.phase2_one_group.native_group_trial",
                    side_effect=lambda layout, selected: {
                        "id": selected["id"] if selected else "baseline",
                        "host_tensor_count": 1 if selected else 0,
                    }),
              patch("calibration.phase2_one_group.probe_native_mapping",
                    side_effect=probe)):
            result = run_one(Path("/fake.gguf"))
        self.assertFalse(result["success"])
        self.assertEqual(result["extended_n_ctx"], 257)
        self.assertTrue(all(mode != "baseline_host_kv" for mode, _ in calls))


if __name__ == "__main__":
    unittest.main()
