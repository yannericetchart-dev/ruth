"""Phase27 offline deterministic API and numerical tests; never call CUDA."""
from __future__ import annotations

import ctypes
import hashlib
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ruth_runtime.gemma3_context_matrix import matrix_cases
from ruth_runtime.gemma3_benchmark_worker import (
    DECODE_STEPS, REPEATS, WARMUPS, evaluate_fixed_sequence,
)
from ruth_runtime.gemma3_benchmark import main, probe_one, verify_logits_artifact
from ruth_runtime.gemma3_logits_compare import compare_logits, read_logits
from ruth_runtime.gemma3_nvtx import NativeRanges


class Phase27GateTests(unittest.TestCase):
    def test_read_only_plan_has_no_worker_or_native_access(self):
        with patch("sys.stdout", new_callable=StringIO) as output, \
             patch("ruth_runtime.gemma3_benchmark.probe_one",
                   side_effect=AssertionError("Never invoke CUDA")):
            self.assertEqual(main([]), 0)
        plan = json.loads(output.getvalue())
        self.assertEqual(len(plan["cases"]), 18)
        self.assertFalse(plan["inference_executed"])
        self.assertFalse(plan["production_qualified"])
        with patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(main(["--case", "3"]), 0)
        self.assertEqual(len(json.loads(output.getvalue())["cases"]), 1)

    def test_gpu_inference_requires_explicit_authorization(self):
        with patch("sys.stderr", new_callable=StringIO):
            for args in (["--execute"], ["--allow-gpu-inference"],
                         ["--execute","--case","2","--start-at","3"]):
                with self.subTest(args=args), self.assertRaises(SystemExit):
                    main(args)
        with patch("ruth_runtime.gemma3_benchmark.inspect_python_environment",
                   side_effect=AssertionError("Unexpected native environment read")):
            for row in matrix_cases():
                result = probe_one(row)
                self.assertEqual(result["status"], "gpu_inference_not_authorized")
                self.assertFalse(result["inference_executed"])
            wrong = dict(matrix_cases()[0], type_k="Q8_0")
            result = probe_one(wrong, authorize_gpu_inference=True)
            self.assertEqual(result["status"], "unapproved_benchmark_configuration")

    def test_invalid_local_logits_artifact_fails_closed(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            self.assertFalse(verify_logits_artifact(root, {
                "relative_path": "../../outside.npz", "sha256": "f"*64}))
            self.assertFalse(verify_logits_artifact(root, {
                "relative_path": ".ruth-build/phase27/gemma-3-1b-it/o.npz",
                "sha256": "0"*64}))


class MockFixedNativeAPI:
    def __init__(self, vocab_size=1024):
        self.vocab_size=vocab_size
        self.calls=[]
        self._values=(ctypes.c_float * vocab_size)(
            *([0.001*float(i) for i in range(vocab_size)]))
    def llama_model_get_vocab(self,model):
        return object()
    def llama_vocab_n_tokens(self,vocab):
        return self.vocab_size
    def llama_tokenize(self,vocab,text,size,buffer,cap,special,parse):
        count = 9
        for i in range(count):
            buffer[i] = 32+i+(100 if special else 0)
        return count
    def llama_get_memory(self,ctx):
        return object()
    def llama_batch_init(self,max_tokens,embedding,seqmax):
        return SimpleNamespace(
            n_tokens=0, token=[0]*max_tokens, pos=[0]*max_tokens,
            n_seq_id=[0]*max_tokens,
            seq_id=[[0] for _ in range(max_tokens)],
            logits=[0]*max_tokens)
    def llama_batch_free(self,batch):
        self.calls.append("free_batch")
    def llama_memory_clear(self,memory,clear):
        self.calls.append("clear")
    def llama_decode(self,ctx,batch):
        self.calls.append((
            "decode", batch.n_tokens,
            [batch.pos[i] for i in range(batch.n_tokens)],
            sum(batch.logits[i] for i in range(batch.n_tokens))))
        return 0
    def llama_synchronize(self,ctx):
        self.calls.append("synchronize")
    def llama_get_logits_ith(self,ctx,index):
        assert index == -1
        return ctypes.cast(self._values,ctypes.POINTER(ctypes.c_float))


class Phase27NumericWorkerTests(unittest.TestCase):
    def test_native_batch_teacher_forcing_and_logits_are_aligned(self):
        native=MockFixedNativeAPI()
        report=evaluate_fixed_sequence(native,object(),object())
        self.assertEqual(report["logits"].shape,(7,1024))
        self.assertEqual(report["logits"].dtype,np.float32)
        self.assertEqual(len(report["prompt_ids"]),9)
        self.assertEqual(len(report["decode_ids"]),6)
        self.assertEqual(len(report["timings"]),3)
        calls=[x for x in native.calls if isinstance(x,tuple)]
        self.assertEqual(len(calls),(WARMUPS+REPEATS)*(DECODE_STEPS+1))
        self.assertEqual(calls[0][1],9)
        self.assertEqual(calls[0][2],list(range(9)))
        self.assertEqual(calls[0][3],1)
        self.assertEqual(calls[1][1:],(1,[9],1))
        self.assertEqual(calls[DECODE_STEPS][2],[14])
        self.assertEqual(native.calls.count("clear"),WARMUPS+REPEATS)
        self.assertEqual(native.calls.count("free_batch"),1)
        self.assertTrue(all(t["prefill_ms"]>0 for t in report["timings"]))


class Phase27NVTXTests(unittest.TestCase):
    def test_missing_nvtx_is_optional_and_never_fakes_dispatch(self):
        with patch("ruth_runtime.gemma3_nvtx.find_library",
                   return_value=None):
            ranges=NativeRanges()
        self.assertFalse(ranges.available)
        with ranges.range("prefill"):
            pass
        with ranges.range("decode"):
            pass
        with self.assertRaises(ValueError):
            with ranges.range("not_attention"):
                pass


class Phase27ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.folder=self.root / ".ruth-build/phase27/gemma-3-1b-it"
        self.folder.mkdir(parents=True)

    def file(self,name,logits=None,**values):
        fp=self.folder/name
        if logits is None:
            logits=np.full((7,1024),0.5,dtype=np.float32)
            logits[:,7]=4.
        data={
            "logits":logits,
            "prompt_ids":np.asarray([3,4,5,6],dtype=np.int32),
            "decode_ids":np.asarray([8,9,10,11,12,13],dtype=np.int32),
        }
        data.update(values)
        np.savez_compressed(fp,**data)
        h=hashlib.sha256(fp.read_bytes()).hexdigest()
        self.assertTrue(verify_logits_artifact(self.root,{
            "relative_path":str(fp.relative_to(self.root)), "sha256":h}))
        return fp

    def test_identical_logits_have_zero_distances(self):
        a=read_logits(self.root,str(self.file("f16.npz")))
        b=read_logits(self.root,str(self.file("bf16.npz")))
        outcome=compare_logits(a,b)
        self.assertEqual(outcome["max_absolute_logit_difference"],0.)
        self.assertEqual(outcome["root_mean_square_logit_difference"],0.)
        self.assertAlmostEqual(outcome["top1_match_fraction"],1.)
        self.assertAlmostEqual(outcome["jensen_shannon_mean_nats"],0.,places=10)
        self.assertFalse(outcome["production_qualified"])

    def test_distribution_changes_are_measured_not_guessed(self):
        p=np.full((7,1024),-2.,dtype=np.float32)
        p[:,7]=2.
        q=p.copy()
        q[:,8]=4.
        a=read_logits(self.root,str(self.file("a.npz",logits=p)))
        b=read_logits(self.root,str(self.file("b.npz",logits=q)))
        outcome=compare_logits(a,b)
        self.assertGreater(outcome["max_absolute_logit_difference"],0)
        self.assertGreater(outcome["root_mean_square_logit_difference"],0)
        self.assertGreater(outcome["kl_baseline_to_candidate_mean_nats"],0)
        self.assertGreater(outcome["jensen_shannon_mean_nats"],0)
        self.assertEqual(outcome["top1_match_fraction"],0.)

    def test_mismatched_tokenization_is_rejected(self):
        a=read_logits(self.root,str(self.file("a.npz")))
        b=read_logits(self.root,str(self.file(
            "b.npz",decode_ids=np.asarray([5,6,7,8,9,10],dtype=np.int32))))
        with self.assertRaisesRegex(ValueError,"Tokenization"):
            compare_logits(a,b)


if __name__=="__main__":
    unittest.main()
