"""Offline tests: native 256 alternative KV and isolated 512 promotion."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from calibration.kv_profile_baseline import run_kv_profile_baseline
from calibration.kv_single_to_512 import run_single_to_512
from tests.test_kv_single_to_512 import make_census, fake_load
from tests.test_kv_adaptive_tensor_offload import fake_native_probe


class AlternativeKVBaselineTests(unittest.TestCase):
    def test_verified_256_native_gpu_kv_allows_separate_512_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            baseline = root / "q8_baseline.json"
            calls = []
            def baseline_probe(path, model_id, trial, **kw):
                calls.append((trial["host_tensor_names"], kw.copy()))
                return fake_native_probe(path, model_id, trial, **kw)
            with patch("calibration.kv_profile_baseline.inspect_gguf",
                       return_value=layout):
                proof = run_kv_profile_baseline(
                    census_path=census, output=baseline,
                    probe=baseline_probe,
                )
            self.assertTrue(proof["success"])
            self.assertTrue(proof["completed"])
            self.assertEqual(proof["status"], "verified_gpu_kv_baseline")
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(
                names == [] and kw["n_ctx"] == 256
                and kw["tokens"] == 16 and kw["flash_attn"] is False
                and kw["type_k"] == "q8_0" and kw["type_v"] == "f16"
                for names, kw in calls
            ))
            self.assertTrue(proof["inference"]["metrics"]["decode_tok_s"] > 0)

            target = root / "q8_stage2.json"
            seen = []
            def load_for_q8(path, model_id, trial, **kwargs):
                seen.append((trial["host_tensor_names"],
                             kwargs["type_k"], kwargs["type_v"]))
                kwargs.pop("flash_attn")
                kwargs.pop("type_k")
                kwargs.pop("type_v")
                return fake_load(path, model_id, trial, **kwargs)

            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                candidate = run_single_to_512(
                    census_path=census, output=target,
                    baseline_profile_path=baseline,
                    flash_attn=False, type_k="q8_0", type_v="f16",
                    load_probe=load_for_q8,
                    inference_probe=fake_native_probe,
                )
            self.assertTrue(candidate["completed"])
            self.assertEqual(candidate["summary"]["candidates"], 6)
            self.assertTrue(candidate["summary"]["inference_success"] >= 1)
            self.assertEqual(candidate["parameters"]["type_k"], "q8_0")
            self.assertIn("baseline_profile_sha256", candidate["parameters"])
            self.assertTrue(all(x[1:] == ("q8_0", "f16") for x in seen))
            self.assertEqual(candidate["summary"]["load_attempted"], 6)
            self.assertEqual(candidate["hard_pruning"]["count"], 0)

    def test_f16_f16_flash_on_256_to_512_has_isolated_outputs(self):
        """FA ON f16/f16 is a distinct native profile, never reused from FA OFF."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            baseline = root / "phase2_f16_f16_faon_baseline_256.json"
            target = root / "phase2_single_to_512_f16_f16_faon.json"
            native_calls = []
            def baseline_probe(path, model_id, trial, **kwargs):
                native_calls.append((list(trial["host_tensor_names"]),
                                     kwargs.copy()))
                return fake_native_probe(path, model_id, trial, **kwargs)
            with patch("calibration.kv_profile_baseline.inspect_gguf",
                       return_value=layout):
                report = run_kv_profile_baseline(
                    census_path=census, output=baseline,
                    flash_attn=True, type_k="f16", type_v="f16",
                    n_ctx=256, probe=baseline_probe,
                )
            self.assertTrue(report["success"])
            self.assertTrue(report["completed"])
            self.assertEqual(len(native_calls), 2)
            self.assertTrue(all(
                names == [] and kwargs["flash_attn"] is True
                and kwargs["type_k"] == kwargs["type_v"] == "f16"
                and kwargs["n_ctx"] == 256 and kwargs["tokens"] == 16
                for names, kwargs in native_calls
            ))
            loaded = []
            inferred = []
            def faon_load(path, model_id, trial, **kwargs):
                loaded.append((tuple(trial["host_tensor_names"]),
                               kwargs.copy()))
                self.assertIs(kwargs.pop("flash_attn"), True)
                self.assertEqual(kwargs.pop("type_k"), "f16")
                self.assertEqual(kwargs.pop("type_v"), "f16")
                return fake_load(path, model_id, trial, **kwargs)
            def faon_inference(path, model_id, trial, **kwargs):
                inferred.append((tuple(trial["host_tensor_names"]),
                                 kwargs.copy()))
                self.assertIs(kwargs["flash_attn"], True)
                self.assertEqual(kwargs["type_k"], "f16")
                self.assertEqual(kwargs["type_v"], "f16")
                return fake_native_probe(path, model_id, trial, **kwargs)
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                promoted = run_single_to_512(
                    census_path=census, output=target,
                    target_ctx=512, flash_attn=True,
                    type_k="f16", type_v="f16",
                    baseline_profile_path=baseline,
                    load_probe=faon_load,
                    inference_probe=faon_inference,
                )
            self.assertTrue(promoted["completed"])
            self.assertEqual(promoted["parameters"]["source_n_ctx"], 256)
            self.assertEqual(promoted["parameters"]["target_n_ctx"], 512)
            self.assertIs(promoted["parameters"]["flash_attn"], True)
            self.assertEqual(promoted["summary"]["inference_success"], 1)
            self.assertEqual(len(loaded), 7)
            self.assertEqual(len(inferred), 2)
            self.assertTrue(all(kw["n_ctx"] == 512 for _, kw in loaded))
            self.assertTrue(all(kw["n_ctx"] == 512 for _, kw in inferred))
            self.assertFalse((root / "phase2_single_to_512.json").exists())
            self.assertEqual(promoted["hard_pruning"]["count"], 0)

    def test_f16_flash_on_promotion_needs_matching_native_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            baseline = root / "f16_faoff_baseline.json"
            with patch("calibration.kv_profile_baseline.inspect_gguf",
                       return_value=layout):
                proof = run_kv_profile_baseline(
                    census_path=census, output=baseline,
                    flash_attn=False, type_k="f16", type_v="f16",
                    probe=fake_native_probe,
                )
            self.assertTrue(proof["success"])
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                with self.assertRaisesRegex(ValueError, "non compatibile"):
                    run_single_to_512(
                        census_path=census,
                        output=root / "should_not_exist.json",
                        target_ctx=512, flash_attn=True,
                        type_k="f16", type_v="f16",
                        baseline_profile_path=baseline,
                    )
            self.assertFalse((root / "should_not_exist.json").exists())

    def test_all_gpu_target_is_probed_before_spending_cpu_offload_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            target = root / "q8_allgpu_512.json"
            attempts = []
            def check(path, model_id, trial, **kwargs):
                attempts.append((list(trial["host_tensor_names"]), kwargs["n_ctx"]))
                return fake_native_probe(path, model_id, trial, **kwargs)
            with patch("calibration.kv_profile_baseline.inspect_gguf",
                       return_value=layout):
                report = run_kv_profile_baseline(
                    census_path=census, output=target,
                    n_ctx=512, probe=check,
                )
            self.assertTrue(report["completed"])
            self.assertFalse(report["success"])
            self.assertEqual(report["status"], "native_baseline_not_verified")
            self.assertEqual(attempts, [([], 512)])
            self.assertEqual(report["parameters"]["n_ctx"], 512)
            self.assertEqual(report["host_tensor_names"], [])

    def test_failure_blocks_promotion_without_claiming_impossibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            baseline = root / "failed.json"
            def fail(path, model_id, trial, **kw):
                return {
                    "success": False,
                    "failure_class": "native_kv_profile_not_verified",
                    "worker_signal": "SIGABRT",
                    "worker_returncode": -6,
                    "stderr_tail": "native GPU KV rejected at 256",
                }
            with patch("calibration.kv_profile_baseline.inspect_gguf",
                       return_value=layout):
                record = run_kv_profile_baseline(
                    census_path=census, output=baseline, probe=fail,
                )
            self.assertTrue(record["completed"])
            self.assertFalse(record["success"])
            self.assertEqual(len(record["samples"]), 1)
            self.assertIn("non", record["note"])
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                with self.assertRaisesRegex(ValueError, "non compatibile"):
                    run_single_to_512(
                        census_path=census,
                        output=root / "must_not_start.json",
                        flash_attn=False, type_k="q8_0", type_v="f16",
                        baseline_profile_path=baseline,
                    )
            self.assertFalse((root / "must_not_start.json").exists())

    def test_alternative_profile_cannot_skip_native_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                with self.assertRaisesRegex(ValueError, "--baseline-profile"):
                    run_single_to_512(
                        census_path=census, output=root / "bad.json",
                        flash_attn=False, type_k="q8_0", type_v="f16",
                    )

    def test_default_f16_f16_checkpoint_identity_is_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout, census = make_census(root)
            with patch("calibration.kv_single_to_512.inspect_gguf",
                       return_value=layout):
                output = run_single_to_512(
                    census_path=census, output=root / "original.json",
                    load_probe=fake_load,
                    inference_probe=fake_native_probe,
                )
            self.assertEqual(output["parameters"]["type_k"], "f16")
            self.assertEqual(output["parameters"]["type_v"], "f16")
            self.assertEqual(output["parameters"]["flash_attn"], False)
            self.assertNotIn("baseline_profile_sha256", output["parameters"])
            self.assertNotIn("ranking_source_note", output["parameters"])


if __name__ == "__main__":
    unittest.main()
