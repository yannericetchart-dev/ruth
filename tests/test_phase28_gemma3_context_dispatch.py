"""Phase28 tests: entirely offline; no CUDA native worker or downloads."""
from __future__ import annotations

import ctypes
from io import StringIO
import json
from pathlib import Path
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ruth_runtime.gemma3_phase28_gate import (
    plans, verify_plan, theoretical_kv_mib, REPEATS, WARMUPS)
from ruth_runtime.gemma3_phase28_worker import (
    run_workload, store_logits, PREFIX, SCHEMA)
from ruth_runtime.gemma3_phase28 import (
    has_memory_reserve, main, probe_one, verify_artifact)
from ruth_runtime.gemma3_phase28_dispatch import inspect_export, classify_kernel


class TestPhase28Gate(unittest.TestCase):
    def test_exactly_four_candidate_types_and_four_context_lengths(self):
        rows = plans()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({(x.candidate,x.n_ctx) for x in rows}),16)
        self.assertEqual({x.n_ctx for x in rows}, {256,512,1024,2048})
        for row in rows:
            self.assertEqual(
                verify_plan(row.candidate,row.n_ctx).as_dict(),row.as_dict())
            self.assertLess(row.prompt_tokens+row.decode_tokens,row.n_ctx)
            self.assertEqual(row.n_batch,64)
            self.assertEqual(row.n_ubatch,64)
            self.assertGreater(theoretical_kv_mib(row),0)
        self.assertEqual(theoretical_kv_mib(verify_plan("f16-off",256)),6.5)
        self.assertEqual(theoretical_kv_mib(verify_plan("f32-off",2048)),104.0)
        for bad in (("f16-off",4096),("unapproved",256),
                    ("bf16-on",1025),("f16-off",True)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                verify_plan(*bad)

    def test_default_is_read_only_and_exact_selection(self):
        with patch("sys.stdout",new_callable=StringIO) as output, \
             patch("ruth_runtime.gemma3_phase28.probe_one",
                   side_effect=AssertionError("Must not access GPU")):
            self.assertEqual(main([]),0)
            data=json.loads(output.getvalue())
            self.assertEqual(data["status"],"read_only_plan")
            self.assertEqual(len(data["cases"]),16)
            self.assertFalse(data["inference_executed"])
        with patch("sys.stdout",new_callable=StringIO) as output:
            self.assertEqual(main(["--candidate","f16-off","--n-ctx","512"]),0)
            self.assertEqual(len(json.loads(output.getvalue())["cases"]),1)

    def test_authorization_required(self):
        with patch("sys.stderr",new_callable=StringIO):
            for flags in (["--execute"], ["--allow-gpu-inference"],
                          ["--execute","--candidate","f16-off","--n-ctx","512"]):
                with self.subTest(flags=flags),self.assertRaises(SystemExit):
                    main(flags)
        with patch("ruth_runtime.gemma3_phase28.inspect_python_environment",
                   side_effect=AssertionError("No environment without authorization")):
            self.assertEqual(probe_one("f16-off",512)["status"],
                             "gpu_inference_not_authorized")
            self.assertEqual(probe_one("f16-off",8192,
                          authorize_gpu_inference=True)["status"],
                             "unapproved_extended_context")

    def test_explicit_free_memory_gate(self):
        self.assertFalse(has_memory_reserve(SimpleNamespace(
            status="gpu_ready",memory_mib={"free":3071})))
        self.assertTrue(has_memory_reserve(SimpleNamespace(
            status="gpu_ready",memory_mib={"free":3072})))
        self.assertFalse(has_memory_reserve(SimpleNamespace(
            status="gpu_busy",memory_mib={"free":6000})))
        self.assertFalse(has_memory_reserve(SimpleNamespace(
            status="gpu_ready",memory_mib=None)))


class FakeNative:
    def __init__(self):
        self.n_vocab=1024
        self.values=(ctypes.c_float * self.n_vocab)(
            *([0.01*float(i) for i in range(self.n_vocab)]))
        self.history=[]
    def llama_model_get_vocab(self,model):
        return object()
    def llama_vocab_n_tokens(self,vocab):
        return self.n_vocab
    def llama_tokenize(self,vocab,text,size,out,capacity,special,parse):
        n=7 if special else 10
        for i in range(n):
            out[i]=100+i
        return n
    def llama_get_memory(self,ctx):
        return object()
    def llama_memory_clear(self,mem,data):
        self.history.append(("clear",))
    def llama_batch_init(self,max_len,embd,seqs):
        return SimpleNamespace(n_tokens=0,token=[0]*max_len,pos=[0]*max_len,
                               seq_id=[[0] for _ in range(max_len)],
                               n_seq_id=[0]*max_len,logits=[0]*max_len)
    def llama_batch_free(self,batch):
        self.history.append(("free",))
    def llama_decode(self,ctx,batch):
        self.history.append((
            "decode",
            batch.n_tokens,
            [batch.pos[i] for i in range(batch.n_tokens)],
            [batch.logits[i] for i in range(batch.n_tokens)]))
        return 0
    def llama_synchronize(self,ctx):
        self.history.append(("sync",))
    def llama_get_logits_ith(self,ctx,i):
        self.history.append(("logits",i))
        return ctypes.cast(self.values,ctypes.POINTER(ctypes.c_float))


class TestPhase28Worker(unittest.TestCase):
    def test_chunked_prefill_and_decode_coordinates(self):
        api=FakeNative()
        plan=verify_plan("bf16-on",512)
        with patch("ruth_runtime.gemma3_nvtx.find_library",return_value=None):
            result=run_workload(api,object(),object(),plan)
        self.assertEqual(result["logits"].shape,(2,1024))
        self.assertEqual(len(result["prompt"]),384)
        self.assertEqual(len(result["decode_ids"]),16)
        self.assertEqual(len(result["timings"]),3)
        self.assertTrue(all(t["prefill_chunks"]==6 for t in result["timings"]))
        dec=[x for x in api.history if x[0]=="decode"]
        self.assertEqual(len(dec),(WARMUPS+REPEATS)*(6+16))
        self.assertEqual(dec[0][2],list(range(64)))
        self.assertEqual(dec[5][2],list(range(320,384)))
        self.assertEqual(sum(dec[5][3]),1)
        self.assertEqual(dec[6][2],[384])
        self.assertEqual(dec[21][2],[399])
        self.assertEqual(api.history.count(("clear",)),4)
        self.assertEqual(api.history.count(("free",)),1)

    def test_local_artifact_sha_and_path_pinning(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            plan=verify_plan("f16-off",256)
            payload={"logits":np.zeros((2,1024),dtype=np.float32),
                     "prompt":list(range(128)),
                     "decode_ids":list(range(16))}
            data=store_logits(root,plan,payload)
            self.assertTrue(verify_artifact(root,data,256))
            self.assertFalse(verify_artifact(root,data,512))
            bad=dict(data,sha256="0"*64)
            self.assertFalse(verify_artifact(root,bad,256))
            self.assertFalse(verify_artifact(root,dict(data,relative_path="../x"),256))


class TestPhase28Dispatch(unittest.TestCase):
    def test_trace_is_read_only_and_never_claims_phase_correlation(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root / ".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"cuda.sqlite"
            with sqlite3.connect(fp) as db:
                db.executescript("""
                    CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT);
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                        (start INTEGER,end INTEGER,demangledName INTEGER);
                    INSERT INTO StringIds VALUES (1,'flash_attn_ext_vec<D=256>');
                    INSERT INTO StringIds VALUES (2,'other_cuda_kernel');
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1,2000001,1);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (4,500004,2);
                """)
            report=inspect_export(fp,root)
            self.assertTrue(report["native_kernel_names_verified"])
            self.assertFalse(report["prefill_decode_attribution_verified"])
            self.assertFalse(report["production_qualified"])
            self.assertEqual(report["kernel_records_total"],2)
            self.assertEqual(report["attention_kernel_records"][0]["attention_family"],
                             "FLASH_VEC")
            with self.assertRaises(ValueError):
                inspect_export(Path("/tmp/not-phase28.sqlite"),root)
        self.assertIsNone(classify_kernel("something unrelated"))


if __name__ == "__main__":
    unittest.main()
