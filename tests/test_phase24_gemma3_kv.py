"""Gemma3 1B KV source-only tests, no live CUDA or downloads."""
import copy
import json
from io import StringIO
from unittest.mock import patch
import unittest

from ruth_runtime.model_profile_v2 import load_profile
from ruth_runtime.gemma3_kv_profile import (
    BACKEND_REVISION, GGUF_SHA256,
    derived_gemma3_geometry, calculate_gemma3_kv)
from ruth_runtime.launcher import main

class Gemma3KVPlanTests(unittest.TestCase):
    def profile(self):
        return load_profile("gemma","gemma-3","gemma-3-1b-it")

    def test_26_layers_6_period_default_partition(self):
        got=derived_gemma3_geometry(self.profile())
        self.assertEqual(got["global_layers"],4)
        self.assertEqual(got["swa_layers"],22)
        self.assertEqual(got["global_layer_indices"],[5,11,17,23])
        self.assertEqual(len(got["swa_layer_indices"]),22)
        self.assertEqual(got["kv_heads"],1)
        self.assertEqual(got["swa_window"],512)
        self.assertEqual(got["llama_cpp_commit"],BACKEND_REVISION)
        self.assertEqual(got["gguf_sha256"],GGUF_SHA256)
        self.assertEqual(got["status"],"pinned_source_derived_native_context_unverified")

    def test_256_f16_f16_baseline_cache_logical_not_cuda(self):
        x=calculate_gemma3_kv()
        self.assertEqual(x["configuration"]["n_ctx_requested"],256)
        self.assertEqual(x["logical_kv"]["global_logical_mib_derived"],1.0)
        self.assertEqual(x["logical_kv"]["swa_logical_mib_derived"],5.5)
        self.assertEqual(x["logical_kv"]["total_logical_mib_derived"],6.5)
        self.assertEqual(x["logical_kv"]["key_logical_mib_derived"],3.25)
        self.assertEqual(x["logical_kv"]["value_logical_mib_derived"],3.25)
        self.assertFalse(x["native_allocation_executed"])
        self.assertFalse(x["gpu_fit_verified"])
        self.assertFalse(x["context_loading_approved"])

    def test_kv_storage_alternatives_are_independent_from_weight_dtypes(self):
        x=calculate_gemma3_kv()
        precision=x["kv_precision_candidates"]
        self.assertTrue(precision["kv_activations_are_not_weight_tensors"])
        self.assertFalse(precision["automatic_cache_dtype_alignment_approved"])
        self.assertFalse(precision["native_context_dtype_support_verified"])
        self.assertEqual(precision["currently_approved_loading_baseline"],
                         {"type_k":"F16","type_v":"F16"})
        costs={k:v["total_logical_mib_derived"]
               for k,v in precision["candidates"].items()}
        self.assertEqual(costs,{"F16":6.5,"BF16":6.5,"F32":13.0})
        # Git already contains the complete GGUF attention tensor inventory.
        self.assertEqual(precision["weight_attention_dtypes"]["attn_k"], {"BF16":26})
        self.assertEqual(precision["weight_attention_dtypes"]["attn_v"], {"BF16":26})
        self.assertEqual(precision["weight_attention_dtypes_status"],
                         "per_layer_attention_weight_formats_observed")

    def test_complete_bf16_attention_weights_are_only_a_kv_hint(self):
        p=copy.deepcopy(self.profile())
        obs=p["architecture"]["gguf_header_observation"]
        obs["attention_projection_coverage"]={
            role:{"layers_present":26,"dtype_counts":{"BF16":26},"complete":True}
            for role in ("attn_q","attn_k","attn_v","attn_output")
        }
        from ruth_runtime.gemma3_kv_profile import kv_precision_candidates
        result=kv_precision_candidates(p,n_ctx=256)
        self.assertEqual(result["weight_attention_dtypes_status"],
                         "per_layer_attention_weight_formats_observed")
        self.assertEqual(result["weight_attention_dtypes"]["attn_k"],{"BF16":26})
        self.assertFalse(result["automatic_cache_dtype_alignment_approved"])
        self.assertFalse(result["candidates"]["BF16"]["native_allocation_verified"])

    def test_aligned_extensions_conserve_per_layer_geometry(self):
        for ctx,mib in ((256,6.5),(512,13.0),(768,19.5),(1024,26.0)):
            x=calculate_gemma3_kv(n_ctx=ctx)
            self.assertEqual(x["logical_kv"]["total_logical_mib_derived"],mib)
        for ctx in (128,257,511):
            with self.assertRaises(ValueError):
                calculate_gemma3_kv(n_ctx=ctx)
        for opts in ({"flash_attention":"auto"},
                     {"swa_full":False},{"type_k":"Q8_0"},
                     {"type_v":"Q8_0"},{"n_batch":64,"n_ubatch":128}):
            with self.assertRaises(ValueError):
                calculate_gemma3_kv(**opts)

    def test_read_only_kv_dtype_pairs_include_mixed_precision(self):
        for k,v,key,value,total,glob,swa,is_baseline in (
            ("F16","F16",3.25,3.25,6.5,1.0,5.5,True),
            ("BF16","BF16",3.25,3.25,6.5,1.0,5.5,False),
            ("F32","F32",6.5,6.5,13.0,2.0,11.0,False),
            ("F32","BF16",6.5,3.25,9.75,1.5,8.25,False),
            ("BF16","F32",3.25,6.5,9.75,1.5,8.25,False),
        ):
            with self.subTest(type_k=k,type_v=v):
                result=calculate_gemma3_kv(type_k=k,type_v=v)
                logical=result["logical_kv"]
                self.assertEqual(logical["key_logical_mib_derived"],key)
                self.assertEqual(logical["value_logical_mib_derived"],value)
                self.assertEqual(logical["total_logical_mib_derived"],total)
                self.assertEqual(logical["global_logical_mib_derived"],glob)
                self.assertEqual(logical["swa_logical_mib_derived"],swa)
                self.assertEqual(result["selected_precision"]["is_current_loading_baseline"],
                                 is_baseline)
                self.assertTrue(result["selected_precision"]["planning_only"])
                self.assertFalse(result["selected_precision"]
                                 ["native_context_dtype_support_verified"])
                self.assertFalse(result["context_loading_approved"])
                self.assertFalse(result["gpu_fit_verified"])
                self.assertFalse(result["native_allocation_executed"])
        self.assertEqual(calculate_gemma3_kv(n_ctx=512,type_k="F32",type_v="BF16")
                         ["logical_kv"]["total_logical_mib_derived"],19.5)

    def test_flash_policy_separates_source_defaults_and_runtime_proof(self):
        off = calculate_gemma3_kv(type_k="BF16",type_v="BF16")
        p = off["kv_attention_policy"]
        self.assertEqual(p["attention_path"],"standard_non_flash")
        self.assertEqual(p["comparison_tier"],"experimental_unverified")
        self.assertIsNone(p["flash_vector_pair_in_pinned_cmake_defaults"])

        for k,v,vector_default,tier in (
            ("F16","F16",True,"reference_candidate"),
            ("BF16","BF16",True,"source_default_vector_candidate"),
            ("F32","F32",False,"conversion_risk_diagnostic"),
            ("BF16","F16",False,"conversion_risk_diagnostic"),
            ("F32","BF16",False,"conversion_risk_diagnostic"),
        ):
            with self.subTest(type_k=k,type_v=v):
                result = calculate_gemma3_kv(
                    flash_attention="on", type_k=k,type_v=v)
                policy = result["kv_attention_policy"]
                self.assertEqual(policy["attention_path"],"flash_dispatch_runtime_unverified")
                self.assertEqual(policy["comparison_tier"],tier)
                self.assertEqual(policy["flash_vector_pair_in_pinned_cmake_defaults"],
                                 vector_default)
                self.assertTrue(policy["flash_f16_conversion_possible"])
                self.assertFalse(policy["installed_cuda_kernel_variants_verified"])
                self.assertFalse(policy["runtime_kernel_dispatch_verified"])
                self.assertFalse(policy["native_context_approved"])
                self.assertFalse(result["context_loading_approved"])
                self.assertFalse(result["native_allocation_executed"])
                self.assertFalse(result["gpu_fit_verified"])
                self.assertFalse(result["selected_precision"]["is_current_loading_baseline"])

        baseline=calculate_gemma3_kv()
        self.assertEqual(baseline["kv_attention_policy"]["comparison_tier"],
                         "reference_baseline")
        self.assertTrue(baseline["selected_precision"]["is_current_loading_baseline"])

    def test_explicit_gguf_swa_pattern_wins_over_backend_default(self):
        profile=copy.deepcopy(self.profile())
        profile["architecture"]["gguf_header_observation"]["metadata_fields"][
            "gemma3.attention.sliding_window_pattern"]=[0]+[1]*25
        x=derived_gemma3_geometry(profile)
        self.assertEqual(x["global_layer_indices"],[0])
        self.assertEqual(x["swa_layers"],25)
        self.assertEqual(x["swa_pattern_source"],"explicit_gguf_layer_array")

    def test_changed_backend_commit_or_missing_gguf_header_fails_closed(self):
        profile=copy.deepcopy(self.profile())
        profile["backend_policies"]["llama.cpp"]["pinned_revision"]="f"*40
        with self.assertRaisesRegex(ValueError,"Unpinned"):
            derived_gemma3_geometry(profile)
        profile=copy.deepcopy(self.profile())
        profile["architecture"].pop("gguf_header_observation",None)
        with self.assertRaisesRegex(ValueError,"not yet observed"):
            derived_gemma3_geometry(profile)

    def test_cli_gemma3_kv_plan_read_only(self):
        with patch("sys.stdout",new_callable=StringIO) as out:
            code=main(["models","kv-plan","--purpose","production",
                       "--family","gemma","--model","gemma-3",
                       "--version","gemma-3-1b-it","--quantization","BF16",
                       "--n-ctx","256","--n-batch","64","--n-ubatch","64",
                       "--swa-full","on","--flash-attn","off",
                       "--type-k","F16","--type-v","F16"])
        self.assertEqual(code,0)
        j=json.loads(out.getvalue())
        self.assertEqual(j["logical_kv"]["total_logical_mib_derived"],6.5)
        self.assertFalse(j["native_allocation_executed"])
        for k,v,mib in (("BF16","BF16",6.5),("F32","F32",13.0),
                         ("BF16","F32",9.75)):
            with self.subTest(type_k=k,type_v=v):
                with patch("sys.stdout",new_callable=StringIO) as out:
                    code=main(["models","kv-plan","--purpose","production",
                               "--family","gemma","--model","gemma-3",
                               "--version","gemma-3-1b-it","--quantization","BF16",
                               "--n-ctx","256","--type-k",k,"--type-v",v])
                self.assertEqual(code,0)
                reply=json.loads(out.getvalue())
                self.assertEqual(reply["logical_kv"]["total_logical_mib_derived"],mib)
                self.assertFalse(reply["context_loading_approved"])
        with patch("sys.stdout",new_callable=StringIO) as out:
            code=main(["models","kv-plan","--purpose","production",
                       "--family","gemma","--model","gemma-3",
                       "--version","gemma-3-1b-it","--quantization","BF16",
                       "--n-ctx","256","--flash-attn","on",
                       "--type-k","BF16","--type-v","BF16"])
        self.assertEqual(code,0)
        on=json.loads(out.getvalue())
        self.assertEqual(on["kv_attention_policy"]["comparison_tier"],
                         "source_default_vector_candidate")
        self.assertFalse(on["context_loading_approved"])

if __name__=="__main__":
    unittest.main()
