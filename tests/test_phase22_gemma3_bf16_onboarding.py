"""New Gemma3 BF16 source onboarding: read-only gate one, no GPU or downloads."""
import json
from io import StringIO
from unittest.mock import patch
import unittest

from ruth_runtime.model_catalog import (
    load_catalog, select_model, resolve_model_metadata,
)
from ruth_runtime.model_profile_v2 import load_profile, profile_path
from ruth_runtime.candidate_selection import load_quantization_policy
from ruth_runtime.launcher import main

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
        self.assertEqual(c["architecture"]["status"],"unknown")
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
