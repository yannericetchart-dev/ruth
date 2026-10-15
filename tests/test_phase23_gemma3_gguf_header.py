"""GGUF metadata/tensor descriptors only; all tests run without CUDA/model weights."""
import base64
import copy
import json
import struct
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from ruth_runtime.gguf_header_inspection import (
    inspect_gguf_header, inspect_pinned_gguf, GGUFInspectError)
from ruth_runtime.model_profile_v2 import load_profile
from ruth_runtime.canonical_profile_git import sync_gguf_architecture
from ruth_runtime.launcher import main

def text(s):
    x=s.encode("utf-8")
    return struct.pack("<Q",len(x))+x

def num(key,value):
    return text(key)+struct.pack("<II",4,value)

def strval(key,value):
    return text(key)+struct.pack("<I",8)+text(value)

def arrayval(key,values):
    return (text(key)+struct.pack("<I",9)+struct.pack("<IQ",4,len(values))
            +struct.pack("<"+"I"*len(values),*values))

def fixture(path:Path, *, kind=24, arch="gemma3", count=26,
            tensor_dtype=30, include_file_type=True,
            complete_first_layer=False, endian="<"):
    attrs=[
        strval("general.architecture",arch),
        *([num("general.file_type",kind)] if include_file_type else []),
        num("gemma3.block_count",count),
        num("gemma3.embedding_length",1152),
        num("gemma3.attention.head_count_kv",1),
        num("gemma3.attention.key_length",256),
        num("gemma3.attention.sliding_window",512),
        arrayval("gemma3.attention.sliding_window_pattern",[1,1,1,1,1,0]),
        text("tokenizer.ggml.tokens")+struct.pack("<I",9)
        +struct.pack("<IQ",8,1200)+b"".join(text("test") for _ in range(1200)),
    ]
    if complete_first_layer:
        tensors = [
            (f"blk.0.{role}.weight",2,1152,1152,30)
            for role in ("attn_q","attn_k","attn_v","attn_output")
        ]
        tensors.append(("blk.0.attn_q_norm.weight",1,256,0,0))
    else:
        tensors = [("blk.0.attn_q.weight",2,1152,1152,tensor_dtype)]
    tensor = b""
    for name,ndim,x,y,dtype in tensors:
        tensor += text(name) + struct.pack("<I",ndim)
        tensor += struct.pack("<Q",x)
        if ndim == 2:
            tensor += struct.pack("<Q",y)
        tensor += struct.pack("<IQ",dtype,0)
    data=b"GGUF"+struct.pack("<IQQ",3,len(tensors),len(attrs))+b"".join(attrs)+tensor
    path.write_bytes(data+b"\x00"*32)


class Gemma3HeaderTests(unittest.TestCase):
    def test_scan_gguf_header_only_skips_tokenizer_and_returns_facts(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"fake.gguf"
            fixture(p)
            report=inspect_gguf_header(p)
        self.assertEqual(report["status"],
                         "gguf_header_geometry_observed_native_crosscheck_pending")
        self.assertEqual(report["gguf_version"],3)
        self.assertEqual(report["tensor_storage_counts"],{"BF16":1})
        self.assertEqual(report["matrix_tensor_elements_by_storage"],
                         {"BF16":1152*1152})
        self.assertEqual(report["declared_file_type"],24)
        self.assertEqual(report["metadata_warnings"],[])
        self.assertEqual(report["metadata_fields"]["gemma3.block_count"],26)
        self.assertEqual(report["metadata_fields"]["gemma3.attention.sliding_window_pattern"],
                         [1,1,1,1,1,0])
        self.assertNotIn("tokenizer.ggml.tokens",report["metadata_fields"])
        self.assertFalse(report["tensor_values_read"])
        self.assertFalse(report["cuda_initialized"])

    def test_attention_role_inventory_reads_weight_formats_not_kv_activations(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"weight_roles.gguf"
            fixture(p,complete_first_layer=True)
            data=inspect_gguf_header(p)
        self.assertEqual(data["tensor_storage_counts"],{"BF16":4,"F32":1})
        self.assertFalse(data["attention_dtypes_are_kv_cache_dtypes"])
        indexed={r["role"]:r for r in data["attention_weight_tensors"]}
        self.assertEqual(set(indexed),{
            "attn_q","attn_k","attn_v","attn_output","attn_q_norm"})
        for r in ("attn_q","attn_k","attn_v","attn_output"):
            self.assertEqual(indexed[r]["dtype"],"BF16")
            self.assertEqual(indexed[r]["shape"],[1152,1152])
            self.assertFalse(data["attention_projection_coverage"][r]["complete"])
        self.assertEqual(indexed["attn_q_norm"]["dtype"],"F32")

    def test_reject_wrong_arch_type_version_and_tensor_dtype(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"x.gguf"
            for kw in ({"arch":"gemma2"},{"count":0},
                       {"tensor_dtype":1}):
                fixture(p,**kw)
                with self.assertRaises(GGUFInspectError):
                    inspect_gguf_header(p)
            p.write_bytes(b"GGUF"+struct.pack("<IQQ",1,1,1))
            with self.assertRaises(GGUFInspectError):
                inspect_gguf_header(p)

    def test_advisory_file_type_mismatch_and_missing_file_type_are_observed(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"x.gguf"
            fixture(p,kind=1)
            v=inspect_gguf_header(p)
            self.assertEqual(v["declared_file_type"],1)
            self.assertEqual(v["weight_format"],"BF16")
            self.assertIn("general.file_type_differs_from_BF16_tensors",
                          v["metadata_warnings"])
            fixture(p,include_file_type=False)
            v=inspect_gguf_header(p)
            self.assertIsNone(v["declared_file_type"])
            self.assertIn("general.file_type_missing",v["metadata_warnings"])

    def test_pinned_inspection_demands_verified_manifest_sha_and_file(self):
        with TemporaryDirectory() as td:
            p=Path(td)/"gguf.gguf"
            fixture(p)
            with patch("ruth_runtime.gguf_header_inspection._valid_local_gguf",
                       return_value=(True,p,None)):
                out=inspect_pinned_gguf(
                    purpose="production",family="gemma",model="gemma-3",
                    version="gemma-3-1b-it",quantization="BF16")
            self.assertTrue(out["source_integrity_verified"])
            self.assertEqual(out["gguf_sha256"],
                             "b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf")
            with patch("ruth_runtime.gguf_header_inspection._valid_local_gguf",
                       return_value=(False,p,"SHA256 mismatch")):
                with self.assertRaisesRegex(GGUFInspectError,"integrity"):
                    inspect_pinned_gguf(
                        purpose="production",family="gemma",model="gemma-3",
                        version="gemma-3-1b-it",quantization="BF16")

    def make_report(self):
        return {
            "status":"gguf_header_geometry_observed_native_crosscheck_pending",
            "gguf_version":3, "architecture":"gemma3", "weight_format":"BF16",
            "tensor_count":1, "tensor_storage_counts":{"BF16":1},
            "matrix_tensor_elements_by_storage":{"BF16":1152*1152},
            "declared_file_type":24, "metadata_warnings":[],
            "tensor_values_read":False,"cuda_initialized":False,
            "model_weights_loaded":False,"context_initialized":False,
            "inference_executed":False, "source_integrity_verified":True,
            "blocks":26,
            "metadata_fields":{
                "general.architecture":"gemma3", "general.file_type":24,
                "gemma3.block_count":26, "gemma3.attention.head_count":4,
                "gemma3.attention.head_count_kv":1,
                "gemma3.attention.key_length":256,
                "gemma3.attention.value_length":256,
                "gemma3.attention.sliding_window":512},
            "model_id":{"family":"gemma","model":"gemma-3",
                        "version":"gemma-3-1b-it"},
            "gguf_sha256":"b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf",
            "gguf_file_bytes":2006573568,
            "source_repository":"bartowski/google_gemma-3-1b-it-GGUF",
            "source_revision":"6de70cb020706f01529b27fee7ab17c5969041c8",
        }

    def test_cas_publication_updates_only_architecture_and_preserves_weights(self):
        model=copy.deepcopy(load_profile("gemma","gemma-3","gemma-3-1b-it"))
        # Simulate a NEW model before the first GGUF-header publication:
        # Git can already contain the verified header from a real host.
        model["architecture"].pop("gguf_header_observation",None)
        model["architecture"]["status"]="unknown"
        original_obs=model["variants"]["BF16"]["runtime_observations"]
        saved={"current":model}
        calls=[]
        def fake(args,**kwargs):
            calls.append(args)
            if args[:4]==["gh","api","user","--jq"]:
                return SimpleNamespace(returncode=0,stdout="yannericetchart-dev",
                                       stderr="")
            if "?ref=model-profiling" in args[-1]:
                b=base64.b64encode(json.dumps(saved["current"]).encode()).decode()
                return SimpleNamespace(returncode=0,
                    stdout=json.dumps({"type":"file","sha":"f"*40,"content":b}),
                    stderr="")
            if args[:3]==["gh","api","--method"]:
                encoded=next(v[8:] for v in args if v.startswith("content="))
                new=json.loads(base64.b64decode(encoded))
                self.assertEqual(new["variants"]["BF16"]["runtime_observations"],
                                 original_obs)
                saved["current"]=new
                return SimpleNamespace(returncode=0,
                    stdout=json.dumps({"content":{"path":"profiles/models/gemma/gemma-3/gemma-3-1b-it/profile.json"},
                                       "commit":{"sha":"f"*40}}),
                    stderr="")
            self.fail("Unexpected gh API arguments")
        report=self.make_report()
        first=sync_gguf_architecture(report,run=fake)
        self.assertTrue(first["git_updated"])
        self.assertEqual(saved["current"]["architecture"]["layer_count"],26)
        self.assertEqual(saved["current"]["architecture"]["kv_heads"],1)
        self.assertEqual(saved["current"]["architecture"]["global_layers"],4)
        self.assertEqual(saved["current"]["architecture"]["swa_layers"],22)
        self.assertEqual(saved["current"]["architecture"]["source_derived_geometry"]
                         ["global_layer_indices"],[5,11,17,23])
        assessment=saved["current"]["architecture"]["kv_precision_assessment_256"]
        self.assertFalse(assessment["automatic_cache_dtype_alignment_approved"])
        self.assertEqual(assessment["candidates"]["F16"]["total_logical_mib_derived"],6.5)
        self.assertEqual(assessment["candidates"]["BF16"]["total_logical_mib_derived"],6.5)
        self.assertEqual(assessment["candidates"]["F32"]["total_logical_mib_derived"],13.0)
        self.assertNotIn("GPU-",json.dumps(saved["current"]))
        second=sync_gguf_architecture(report,run=fake)
        self.assertEqual(second["status"],"git_profile_already_current")
        writes=[x for x in calls if x[:3]==["gh","api","--method"]]
        self.assertEqual(len(writes),1)

    def test_advisory_mismatch_can_be_published_with_provenance_warning(self):
        report=self.make_report()
        report["metadata_fields"]["general.file_type"]=1
        report["declared_file_type"]=1
        report["metadata_warnings"]=[
            "general.file_type_differs_from_BF16_tensors"]
        profile=copy.deepcopy(load_profile("gemma","gemma-3","gemma-3-1b-it"))
        profile["architecture"].pop("gguf_header_observation",None)
        profile["architecture"]["status"]="unknown"
        seen={"current":profile}
        def fake(args,**kwargs):
            if args[:4]==["gh","api","user","--jq"]:
                return SimpleNamespace(returncode=0,stdout="yannericetchart-dev",
                                       stderr="")
            if "?ref=model-profiling" in args[-1]:
                blob=base64.b64encode(json.dumps(seen["current"]).encode()).decode()
                return SimpleNamespace(returncode=0,
                   stdout=json.dumps({"type":"file","sha":"f"*40,"content":blob}),
                   stderr="")
            if args[:3]==["gh","api","--method"]:
                encoded=next(v[8:] for v in args if v.startswith("content="))
                seen["current"]=json.loads(base64.b64decode(encoded))
                return SimpleNamespace(returncode=0,
                    stdout=json.dumps({"content":{"path":"profiles/models/gemma/gemma-3/gemma-3-1b-it/profile.json"},
                                       "commit":{"sha":"f"*40}}),stderr="")
            self.fail("Unexpected GitHub arguments")
        result=sync_gguf_architecture(report,run=fake)
        self.assertTrue(result["git_updated"])
        fact=seen["current"]["architecture"]["gguf_header_observation"]
        self.assertEqual(fact["declared_file_type"],1)
        self.assertIn("general.file_type_differs_from_BF16_tensors",
                      fact["metadata_warnings"])

    def test_prior_verified_header_may_be_enriched_without_erasing_native_weights(self):
        # A real owner has already published the base GGUF header. Adding
        # later per-layer tensor roles must preserve all original fields.
        from collections import Counter
        initial=self.make_report()
        old={
            "status":"gguf_header_observed_native_crosscheck_pending",
            "gguf_sha256":initial["gguf_sha256"],
            "gguf_version":initial["gguf_version"],
            "tensor_count":initial["tensor_count"],
            "tensor_storage_counts":initial["tensor_storage_counts"],
            "matrix_tensor_elements_by_storage":initial["matrix_tensor_elements_by_storage"],
            "declared_file_type":initial["declared_file_type"],
            "metadata_warnings":initial["metadata_warnings"],
            "metadata_fields":initial["metadata_fields"],
        }
        original=copy.deepcopy(load_profile("gemma","gemma-3","gemma-3-1b-it"))
        original["architecture"]["gguf_header_observation"]=old
        original["architecture"]["source_derived_geometry"]=None
        record=copy.deepcopy(initial)
        record["attention_dtypes_are_kv_cache_dtypes"]=False
        record["attention_weight_tensors"]=[
            {"layer":0,"role":"attn_q","dtype":"BF16","shape":[1152,1152]}
        ]
        record["attention_projection_coverage"]={
            role:{
                "layers_present":1 if role=="attn_q" else 0,
                "dtype_counts":{"BF16":1} if role=="attn_q" else {},
                "complete":False,
            } for role in ("attn_q","attn_k","attn_v","attn_output")
        }
        state={"value":original}
        def fake(args,**kw):
            if args[:4]==["gh","api","user","--jq"]:
                return SimpleNamespace(returncode=0,stdout="yannericetchart-dev",stderr="")
            if "?ref=model-profiling" in args[-1]:
                blob=base64.b64encode(json.dumps(state["value"]).encode()).decode()
                return SimpleNamespace(returncode=0,
                   stdout=json.dumps({"type":"file","sha":"f"*40,"content":blob}),stderr="")
            if args[:3]==["gh","api","--method"]:
                payload=next(v[8:] for v in args if v.startswith("content="))
                state["value"]=json.loads(base64.b64decode(payload))
                return SimpleNamespace(returncode=0,
                    stdout=json.dumps({"content":{"path":"profiles/models/gemma/gemma-3/gemma-3-1b-it/profile.json"},
                                       "commit":{"sha":"f"*40}}),stderr="")
            self.fail("Unexpected GitHub request")
        result=sync_gguf_architecture(record,run=fake)
        self.assertTrue(result["git_updated"])
        actual=state["value"]
        self.assertEqual(actual["architecture"]["gguf_header_observation"]["metadata_fields"],old["metadata_fields"])
        self.assertFalse(actual["architecture"]["gguf_header_observation"]["attention_dtypes_are_kv_cache_dtypes"])
        self.assertEqual(actual["architecture"]["gguf_header_observation"]["attention_weight_tensors"],
                         record["attention_weight_tensors"])
        self.assertFalse(actual["architecture"]["kv_precision_assessment_256"]
                         ["automatic_cache_dtype_alignment_approved"])
        self.assertEqual(actual["architecture"]["kv_precision_assessment_256"]
                         ["weight_attention_dtypes_status"],
                         "per_layer_attention_weight_formats_unknown")
        self.assertEqual(actual["variants"]["BF16"]["runtime_observations"],
                         original["variants"]["BF16"]["runtime_observations"])
        second=sync_gguf_architecture(record,run=fake)
        self.assertEqual(second["status"],"git_profile_already_current")

    def test_reject_unpinned_geometry_before_git_login(self):
        r=self.make_report()
        r["gguf_sha256"]="f"*64
        with self.assertRaises(ValueError):
            sync_gguf_architecture(r,run=lambda *a,**kw: self.fail("no Git"))

    def test_cli_gguf_inspector_no_git_writes_if_opted_out(self):
        fake=self.make_report()
        with (patch("ruth_runtime.gguf_header_inspection.inspect_pinned_gguf",
                    return_value=fake),
              patch("ruth_runtime.canonical_profile_git.sync_gguf_architecture",
                    side_effect=AssertionError("Git forbidden")),
              patch("sys.stdout",new_callable=StringIO) as output):
            status=main(["models","inspect-gguf","--family","gemma",
                "--model","gemma-3","--version","gemma-3-1b-it",
                "--quantization","BF16","--no-git-sync"])
        self.assertEqual(status,0)
        result=json.loads(output.getvalue())
        self.assertFalse(result["runtime_profile_published"])
        self.assertFalse(result["tensor_values_read"])
        self.assertEqual(result["git_publication"]["status"],"git_sync_disabled")


if __name__=="__main__":
    unittest.main()
