"""No-GPU regression tests for cumulative single GGUF tensor census."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.kv_adaptive_tensor_offload import (
    _probe_summary, summarize_probe_repeats,
)
from calibration.kv_single_tensor_census import (
    _from_existing, _native_verified, run_single_tensor_census,
)
from calibration.mapping_grouping import native_tensor_trial
from tests.test_kv_adaptive_tensor_offload import (
    fake_native_probe, layout_fixture,
)


def verified_trial(layout, model, names, *, ctx=256, tokens=16, repeats=2):
    trial = native_tensor_trial(layout, names)
    samples = [
        _probe_summary(fake_native_probe(
            model, "gemma-2-9b-it", trial, n_ctx=ctx,
            tokens=tokens, timeout=600, flash_attn=False,
            type_k="f16", type_v="f16",
        ))
        for _ in range(repeats)
    ]
    result = summarize_probe_repeats(samples, repeats)
    result.update({
        "n_ctx": ctx, "host_tensor_names": sorted(names),
        "mapping_id": trial["id"],
    })
    if result["success"]:
        metrics = result["metrics"]
        metrics["combined_time_s"] = round(
            (ctx-tokens)/metrics["prefill_tok_s"]
            + tokens/metrics["decode_tok_s"], 6
        )
    return result


def prepare(root):
    model = root / "model.gguf"
    model.write_bytes(b"model placeholder")
    layout = layout_fixture()
    base = verified_trial(layout, model, [])
    original = verified_trial(layout, model, ["blk.0.ffn_down.weight"])
    source = {
        "schema": "ruth-phase2-adaptive-tensor-greedy-v3",
        "status": "confirmed_all_requested_contexts",
        "completed": True,
        "actual_profiling_start_ctx": 256,
        "parameters": {
            "model_path": str(model.resolve()),
            "model_id": "gemma-2-9b-it",
            "generated_tokens": 16,
            "scouting_repeats": 2,
            "baseline_placement_protocol":
                "exact_phase1_with_native_kv_step_fallback_v3",
        },
        "stages": [{"successful": True, "from_n_ctx": 256}],
        "measured_configs": {
            "baseline": base, "single": original,
            "unused_pair": verified_trial(
                layout, model,
                ["blk.0.ffn_down.weight", "blk.1.ffn_down.weight"],
            ),
        },
    }
    source_path = root / "greedy_v3.json"
    source_path.write_text(json.dumps(source))
    return model, layout, source_path


class SingleTensorCensusTests(unittest.TestCase):
    def test_all_six_reuse_one_run_other_five_without_repeating(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, layout, source = prepare(root)
            path = root / "census.json"
            calls = []
            def observe(*args, **kwargs):
                calls.append(tuple(args[2]["host_tensor_names"]))
                return fake_native_probe(*args, **kwargs)
            with patch("calibration.kv_single_tensor_census.inspect_gguf",
                       return_value=layout):
                report = run_single_tensor_census(
                    source_report=source, output=path, probe=observe,
                )
            self.assertTrue(report["completed"])
            self.assertEqual(report["summary"]["total_eligible"], 6)
            self.assertEqual(report["summary"]["source_reused_success"], 1)
            self.assertEqual(report["summary"]["new_success"], 5)
            self.assertEqual(report["summary"]["verified_total"], 6)
            self.assertEqual(len(calls), 10)
            self.assertNotIn(("blk.0.ffn_down.weight",), calls)
            self.assertTrue(all(
                x.get("marginal", {}).get("gain_mib") is not None
                for x in report["tensor_results"].values()
            ))
            self.assertEqual(
                json.loads(source.read_text())["measured_configs"]["baseline"][
                    "host_tensor_names"], []
            )

    def test_checkpoint_resume_with_incremented_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, layout, source = prepare(root)
            path = root / "checkpoint.json"
            calls = []
            def observe(*args, **kwargs):
                calls.append(tuple(args[2]["host_tensor_names"]))
                return fake_native_probe(*args, **kwargs)
            with patch("calibration.kv_single_tensor_census.inspect_gguf",
                       return_value=layout):
                first = run_single_tensor_census(
                    source_report=source, output=path,
                    max_new_tests=2, probe=observe,
                )
                self.assertFalse(first["completed"])
                self.assertEqual(first["summary"]["verified_total"], 3)
                self.assertEqual(first["summary"]["missing_count"], 3)
                saved = set(first["new_measurements"])
                second = run_single_tensor_census(
                    source_report=source, output=path,
                    max_new_tests=3, probe=observe,
                )
            self.assertTrue(second["completed"])
            self.assertEqual(len(calls), 10)
            self.assertTrue(saved.issubset(second["new_measurements"]))
            self.assertEqual(second["summary"]["verified_total"], 6)

    def test_native_failure_never_counted_as_verified_retry_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, layout, source = prepare(root)
            path = root / "failures.json"
            bad_name = "blk.0.attn_k.weight"
            def fail_specific(*args, **kwargs):
                result = fake_native_probe(*args, **kwargs)
                if args[2]["host_tensor_names"] == [bad_name]:
                    result["success"] = False
                    result["failure_class"] = "native_placement_unverified"
                    result["gpu_runtime_validation"]["verified"] = False
                return result
            with patch("calibration.kv_single_tensor_census.inspect_gguf",
                       return_value=layout):
                failed = run_single_tensor_census(
                    source_report=source, output=path, probe=fail_specific,
                )
                self.assertFalse(failed["completed"])
                self.assertEqual(failed["summary"]["failed_count"], 1)
                self.assertEqual(failed["summary"]["missing_count"], 0)
                self.assertIn(bad_name, failed["summary"]["failed_tensors"])

                # Existing unsuccessful measurement is left intact until
                # explicitly opted in for a repeat CUDA trial.
                calls = []
                def counted(*args, **kwargs):
                    calls.append(tuple(args[2]["host_tensor_names"]))
                    return fake_native_probe(*args, **kwargs)
                again = run_single_tensor_census(
                    source_report=source, output=path, probe=counted,
                )
                self.assertFalse(again["completed"])
                self.assertEqual(calls, [])
                repaired = run_single_tensor_census(
                    source_report=source, output=path,
                    retry_failures=True, probe=counted,
                )
                self.assertTrue(repaired["completed"])
                self.assertEqual(calls, [(bad_name,), (bad_name,)])

    def test_reject_cache_with_forged_gpu_kv_or_wrong_token_shape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model, layout, source = prepare(root)
            row = verified_trial(
                layout, model, ["blk.0.ffn_down.weight"]
            )
            kwargs = {
                "n_ctx": 256,
                "names": ["blk.0.ffn_down.weight"],
                "trial_id": row["mapping_id"],
                "repeats": 2, "tokens": 16,
            }
            self.assertTrue(_native_verified(row, **kwargs))
            wrong = copy.deepcopy(row)
            wrong["samples"][0]["gpu_kv_verified"] = False
            self.assertFalse(_native_verified(wrong, **kwargs))
            wrong = copy.deepcopy(row)
            wrong["samples"][0]["prompt_tokens"] = 239
            self.assertFalse(_native_verified(wrong, **kwargs))

    def test_reject_changed_source_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, layout, source = prepare(root)
            with patch("calibration.kv_single_tensor_census.inspect_gguf",
                       return_value=layout):
                with self.assertRaisesRegex(ValueError, "sovrascrivere"):
                    run_single_tensor_census(
                        source_report=source, output=source,
                        max_new_tests=1,
                    )
                out = root / "census.json"
                run_single_tensor_census(
                    source_report=source, output=out,
                    max_new_tests=1, probe=fake_native_probe,
                )
                data = json.loads(source.read_text())
                data["parameters"]["generated_tokens"] = 8
                source.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "Checkpoint incompatibile"):
                    run_single_tensor_census(
                        source_report=source, output=out,
                        max_new_tests=1, probe=fake_native_probe,
                    )


if __name__ == "__main__":
    unittest.main()
