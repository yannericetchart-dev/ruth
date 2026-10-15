"""New Gemma3 BF16 source onboarding: read-only gate one, no GPU or downloads."""
import json
from io import StringIO
from unittest.mock import patch
import unittest

from ruth_runtime.model_catalog import (
    load_catalog, select_model, resolve_model_metadata,
)
from ruth_runtime.model_profile_v2 import load_profile, profile_path
from ruth_runtime.candidate_selection import load_quantization_policy, discover_quantizations
from ruth_runtime.launcher import main
from ruth_runtime.load_only_probe import probe_model_load
from ruth_runtime.load_only_worker import APPROVED_LOADS

REV="6de70cb020706f01529b27fee7ab17c5969041c8"
DIGEST="b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf"
SIZE=2006573568


class Gemma3Bf16OnboardingTests(unittest.TestCase):
    def choice(self,purpose="production"):
        return select_model(
            purpose,family="gemma",model="gemma-3",
            version="gemma-3-1b-it",format="gguf",quantization="BF16")

    def test_pinned_model_matches_hf_file_link(self):
        c=self.choice()
        self.assertEqual(c.repo_id,"bartowski/google_gemma-3-1b-it-GGUF")
        self.assertEqual(c.filename,"google_gemma-3-1b-it-bf16.gguf")
        self.assertEqual(c.expected_sha256,DIGEST)
        self.assertEqual(c.expected_size_bytes,SIZE)
        self.assertEqual(c.revision,REV)
        self.assertEqual(c.license,"gemma")
        self.assertTrue(c.license_acknowledgement_required)

    def test_same_canonical_profile_from_production_or_testing(self):
        a=self.choice("production")
        b=self.choice("test")
        self.assertEqual(a.expected_sha256,b.expected_sha256)
        c=load_profile("gemma","gemma-3","gemma-3-1b-it")
        self.assertEqual(c["variants"]["BF16"]["precision_family"],"floating")
        self.assertEqual(c["variants"]["BF16"]["file_bytes"],SIZE)
        self.assertEqual(c["selection"]["original_checkpoint_dtype"],"BF16")
        self.assertIn(c["architecture"]["status"], (
            "gguf_header_observed_native_crosscheck_pending",
            "gguf_header_and_pinned_backend_source_derived_native_context_unverified"))
        self.assertEqual(c["architecture"]["layer_count"],26)
        self.assertEqual(c["architecture"]["kv_heads"],1)
        if "source_derived_geometry" in c["architecture"]:
            self.assertEqual(c["architecture"]["global_layers"], 4)
            self.assertEqual(c["architecture"]["swa_layers"], 22)
        else:
            self.assertIsNone(c["architecture"]["global_layers"])
        self.assertEqual(c["variants"]["BF16"]["context_observations"],{})
        self.assertEqual(profile_path("gemma","gemma-3","gemma-3-1b-it").name,
                         "profile.json")

    def test_model_metadata_is_strict_and_network_not_automatic(self):
        c=self.choice()
        denied=resolve_model_metadata(c)
        self.assertEqual(denied["status"],"metadata_network_not_authorized")
        upstream={"sha":REV,"private":False,"gated":False,
                  "cardData":{"license":"gemma"},
                  "siblings":[{"rfilename":c.filename,
                               "size":SIZE,
                               "lfs":{"sha256":DIGEST,"size":SIZE}}]}
        checked=resolve_model_metadata(
            c,authorize_metadata_network=True,
            fetch=lambda repo:upstream)
        self.assertEqual(checked["status"],
                         "source_metadata_verified_license_pending")
        self.assertFalse(checked["download_started"])
        self.assertFalse(checked["license_accepted"])
        self.assertFalse(checked["gpu_offload_executed"])
        tampered=json.loads(json.dumps(upstream))
        tampered["siblings"][0]["lfs"]["sha256"]="f"*64
        rejected=resolve_model_metadata(
            c,authorize_metadata_network=True,fetch=lambda repo:tampered)
        self.assertEqual(rejected["status"],"source_metadata_unverified")

    def test_quality_class_and_float_subpriority_are_separate(self):
        p=load_quantization_policy()
        floating={r["quantization"]:r for r in p["quality_policy"]
                  if r["quantization"] in ("BF16","F16","F32")}
        self.assertTrue(all(r["quality_class"]==1 for r in floating.values()))
        self.assertEqual([floating[f]["floating_subpriority"]
                          for f in ("BF16","F16","F32")],[1,2,3])
        self.assertGreater(floating["BF16"]["priority"],floating["F16"]["priority"])
        self.assertGreater(floating["F16"]["priority"],floating["F32"]["priority"])
        self.assertIsNone(floating["BF16"]["numeric_fidelity_warning"])
        self.assertIn("overflow",floating["F16"]["numeric_fidelity_warning"])
        self.assertIn("Tensor Core",floating["F32"]["numeric_fidelity_warning"])
        self.assertNotIn("BF32",[x["quantization"] for x in p["quality_policy"]])

    def test_selected_gemma3_inventory_handles_lowercase_bf16_gguf_name(self):
        choice=self.choice()
        raw={"sha":REV,"private":False,"gated":False,
             "cardData":{"license":"gemma"},
             "siblings":[{"rfilename":choice.filename,"size":SIZE,
                          "lfs":{"sha256":DIGEST,"size":SIZE}}]}
        inv=discover_quantizations(purpose="production",
                 authorize_metadata_network=True,selected_choice=choice,
                 fetch=lambda repo:raw)
        self.assertEqual(inv["status"],"quantizations_discovered")
        self.assertEqual(inv["model_path"],"gemma/gemma-3/gemma-3-1b-it/gguf")
        self.assertEqual(len(inv["quantizations"]),1)
        self.assertEqual(inv["quantizations"][0]["quantization"],"bf16")
        self.assertTrue(inv["quantizations"][0]["catalog_pin_exists"])
        self.assertTrue(inv["quantizations"][0]["download_pin_ready"])
        self.assertFalse(inv["download_started"])

    def test_gemma3_bf16_context_requires_exact_pinned_experiment(self):
        self.assertEqual(APPROVED_LOADS["gemma-3-1b-it"]["quantizations"],("BF16",))
        self.assertEqual(APPROVED_LOADS["gemma-3-1b-it"]["modes"],("weights","context"))
        with patch("ruth_runtime.load_only_probe.inspect_python_environment",
                   side_effect=AssertionError("Do not inspect CUDA environment")):
            x=probe_model_load(model_version="gemma-3-1b-it",
                               quantization="BF16",mode="context",
                               allow_gpu_load=True,n_ctx=256,
                               flash_attention="auto")
        self.assertEqual(x["status"],"unsupported_kv_context_configuration")
        self.assertFalse(x["weights_loaded"])
        with patch("sys.stdout",new_callable=StringIO) as out:
            rc=main(["models","probe-load","--purpose","production",
                     "--family","gemma","--model","gemma-3",
                     "--version","gemma-3-1b-it","--quantization","BF16",
                     "--load-mode","weights"])
        self.assertEqual(rc,2)
        self.assertEqual(json.loads(out.getvalue())["status"],
                         "gpu_load_not_authorized")

    def test_cli_select_gemma3_offline_does_not_download_or_infer(self):
        with (patch("sys.stdout",new_callable=StringIO) as out,
              patch("ruth_runtime.hardware_eligibility.inspect_host_hardware",
                    return_value={"status":"hardware_unverified"})):
            code=main(["models","select","--purpose","production",
                       "--family","gemma","--model","gemma-3",
                       "--version","gemma-3-1b-it",
                       "--quantization","BF16"])
        self.assertEqual(code,0)
        x=json.loads(out.getvalue())
        self.assertEqual(x["selection"]["expected_sha256"],DIGEST)
        self.assertEqual(x["canonical_profile"]["status"],"canonical_profile_loaded")
        self.assertEqual(x["canonical_profile"]["known_runtime_configurations"],0)
        self.assertFalse(x["download_started"])
        self.assertFalse(x["gpu_allocations_executed"])


if __name__=="__main__":
    unittest.main()
