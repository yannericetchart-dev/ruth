"""Canonical model profiles, immutable source pins and 256-step load planning.

Uses pinned, committed Git records. No CUDA, model download or network access.
"""
import json
import shutil
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from ruth_runtime.dependencies import PROJECT_ROOT
from ruth_runtime.model_catalog import load_catalog
from ruth_runtime.model_profile_v2 import (
    create_profile, evaluate_loading_fit, load_probe_config, load_profile,
    merge_published_measurement, resolve_model_profile, profile_path)
from ruth_runtime.candidate_selection import load_quantization_policy
from ruth_runtime.launcher import main
from ruth_runtime.production_selection import _published_minimal_inference
from ruth_runtime.offline_memory_fit import explain_offline_weights_fit


class CanonicalProfilesTests(unittest.TestCase):
    def test_gemma_has_exactly_one_canonical_profile_in_git(self):
        p = profile_path("gemma", "gemma-2", "gemma-2-9B-it")
        self.assertTrue(p.is_file())
        v = load_profile("gemma", "gemma-2", "gemma-2-9B-it")
        self.assertEqual(v["schema"], "ruth-model-profile-v2")
        self.assertEqual(set(v["variants"]), {"Q4_K_L", "Q4_K_M"})
        self.assertEqual(v["architecture"]["global_layers"], 21)
        self.assertEqual(v["architecture"]["swa_layers"], 21)
        self.assertTrue(v["provenance"]["historical_migration_completed"])
        self.assertEqual(v["backend_policies"]["llama.cpp"]["gguf_min_n_ctx_requested"],256)
        self.assertEqual(v["backend_policies"]["llama.cpp"]["context_step"],256)
        self.assertEqual(set(v["backend_policies"]["llama.cpp"]["weight_types"]),
                         {"F32", "F16", "BF16"})

    def test_historical_off_and_on_retained_without_becoming_256_load_proof(self):
        v = load_profile("gemma", "gemma-2", "gemma-2-9B-it")["variants"]["Q4_K_M"]
        self.assertTrue(v["runtime_observations"])
        profiles = [x["data"] for bucket in v["context_observations"].values()
                    for x in bucket["observations"]]
        modes = {r.get("flash_attention") for r in profiles}
        self.assertIn("off", modes)
        self.assertIn("on", modes)
        self.assertEqual(len([r for r in profiles if r.get("flash_attention") == "off"
                              and r.get("kv_type_k") == "F16"]), 1)
        self.assertTrue(all(r["n_ctx_requested"] == 128 for r in profiles))
        same = evaluate_loading_fit(load_profile("gemma", "gemma-2", "gemma-2-9B-it"),
                                    quantization="Q4_K_M", host={
                                        "total_mib":6141, "architecture":"ada",
                                        "backend_cuda":"13.2", "driver_version":"595.99.02",
                                        "llama_cpp_binding":"0.3.36"},
                                    n_ctx=256)
        self.assertEqual(same["status"], "exact_loading_context_unverified")

    def test_historical_published_v1_is_idempotent_for_canonical_merge(self):
        root = PROJECT_ROOT / "profiles/measurements"
        p = next((root / "13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe").glob(
            "9c3654857ded1eafe9a8fa41198d3063e49a339ac6dd57e10359a07adea9ac97.json"))
        payload = json.loads(p.read_text())
        merged, changed = merge_published_measurement(
            load_profile("gemma", "gemma-2", "gemma-2-9B-it"),payload)
        self.assertFalse(changed)
        self.assertEqual(merged["variants"]["Q4_K_M"]["context_observations"],
                         load_profile("gemma", "gemma-2", "gemma-2-9B-it")
                         ["variants"]["Q4_K_M"]["context_observations"])

    def test_missing_model_profile_can_be_created_without_gpu(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"catalog").mkdir()
            shutil.copyfile(PROJECT_ROOT/"catalog/model_catalog.json",
                            root/"catalog/model_catalog.json")
            a=resolve_model_profile("production","gemma","gemma-2",
                                    "gemma-2-9B-it",project_root=root)
            b=resolve_model_profile("test","gemma","gemma-2",
                                    "gemma-2-9B-it",project_root=root)
            self.assertEqual(a["status"],"canonical_profile_created_in_memory")
            self.assertEqual(a["profile"],b["profile"])
            self.assertFalse(a["gpu_allocations_executed"])
            self.assertIsNone(a["profile"]["backend_policies"]["llama.cpp"]
                              ["n_ctx_256_profile_batch"])
            self.assertEqual(a["profile"]["variants"]["Q4_K_M"]
                             ["context_observations"],{})

    def test_canonical_only_git_profile_reuses_native_token_and_weight_evidence(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"catalog").mkdir()
            for name in ("model_catalog.json","nvidia_architecture_policy.json"):
                shutil.copyfile(PROJECT_ROOT/"catalog"/name,root/"catalog"/name)
            entry=profile_path("gemma","gemma-2","gemma-2-9B-it")
            copy=root/"profiles/models/gemma/gemma-2/gemma-2-9B-it/profile.json"
            copy.parent.mkdir(parents=True)
            shutil.copyfile(entry,copy)
            # No profiles/measurements archive exists in this test root.
            hw={"architecture":"ada","driver_version":"595.99.02",
                "backend_cuda":"13.2"}
            verified=_published_minimal_inference(
                root=root,sha="13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe",
                hardware=hw)
            self.assertEqual(verified["n_ctx"],128)
            self.assertIn("profiles/models/gemma/",verified["source"])
            fit=explain_offline_weights_fit(
                project_root=root,quantization="Q4_K_M",
                architecture="ada",vram_gib=6,driver_version="595.99.02",
                backend_cuda="13.2",binding_version="0.3.36")
            self.assertEqual(fit["status"],"weights_possible_context_unverified")
            self.assertEqual(fit["memory"]["weights_cuda_buffer_mib"],5488.4)
            self.assertEqual(len(fit["evidence_files"]),1)
            self.assertIn("profiles/models/gemma/",fit["evidence_files"][0])

    def test_positive_host_reuse_does_not_promote_legacy_128_to_256(self):
        profile=load_profile("gemma","gemma-2","gemma-2-9B-it")
        host={"total_mib":6141,"architecture":"ada","backend_cuda":"13.2",
              "driver_version":"595.99.02","llama_cpp_binding":"0.3.36"}
        old=evaluate_loading_fit(profile,quantization="Q4_K_M",
                                 host=host,n_ctx=128)
        self.assertEqual(old["status"],"exact_loading_context_previously_observed")
        self.assertEqual(old["matched_context"]["gpu_kv_buffer_mib"],84.0)
        new=evaluate_loading_fit(profile,quantization="Q4_K_M",host=host,n_ctx=256)
        self.assertEqual(new["status"],"exact_loading_context_unverified")

    def test_requested_context_must_start_256_and_step_256(self):
        v=load_profile("gemma","gemma-2","gemma-2-9B-it")
        for c in (256,512,768,1024):
            self.assertEqual(load_probe_config(v,n_ctx=c,n_batch=64,n_ubatch=64)["n_ctx"],c)
        for c in (1,128,257,511):
            with self.assertRaises(ValueError):
                load_probe_config(v,n_ctx=c,n_batch=64,n_ubatch=64)
        with self.assertRaises(ValueError):
            load_probe_config(v,n_ctx=256,n_batch=256,n_ubatch=32,type_v="Q8_0")

    def test_f32_f16_bf16_recognized_without_claiming_better_model_quality(self):
        policy=load_quantization_policy()
        q={v["quantization"] for v in policy["quality_policy"]}
        self.assertTrue({"F32","F16","BF16","f32","f16","bf16"}.issubset(q))
        e=next(x for x in load_catalog()["models"] if x["version"]=="gemma-2-9B-it")
        candidate=create_profile(e)
        self.assertIsNone(candidate["selection"]["cross_model_quality_score"])

    def test_select_for_production_or_testing_reuses_identical_model_profile(self):
        with patch("sys.stdout",new_callable=StringIO) as out:
            code=main(["models","select","--purpose","production",
                       "--quantization","Q4_K_M"])
        self.assertEqual(code,0)
        p=json.loads(out.getvalue())
        self.assertEqual(p["canonical_profile"]["status"],"canonical_profile_loaded")
        self.assertGreater(p["canonical_profile"]["known_runtime_configurations"],0)
        with patch("sys.stdout",new_callable=StringIO) as out:
            code=main(["models","select","--purpose","test",
                       "--family","gemma","--model","gemma-2",
                       "--version","gemma-2-9B-it","--quantization","Q4_K_M"])
        self.assertEqual(code,0)
        v=json.loads(out.getvalue())
        self.assertEqual(p["canonical_profile"]["path"],v["canonical_profile"]["path"])

    def test_probe_cli_rejects_128_without_loading_cuda(self):
        with patch("ruth_runtime.load_only_probe.probe_model_load",
                   side_effect=AssertionError("Do not load GPU")), \
             patch("sys.stdout",new_callable=StringIO) as out:
            status=main(["models","probe-load","--quantization","Q4_K_M",
                         "--load-mode","context","--n-ctx","128",
                         "--allow-gpu-load"])
        self.assertEqual(status,2)
        self.assertNotEqual(json.loads(out.getvalue()).get("status"),"gpu_load_only_completed")


if __name__=="__main__":
    unittest.main()
