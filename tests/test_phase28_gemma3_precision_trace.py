"""Synthetic Nsight node-level precision name survey: no CUDA."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_precision_trace import (
    classify_conversion, inspect_precision_trace)


def write_nsight(file: Path):
    with sqlite3.connect(file) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,end INTEGER,eventType INTEGER,
                globalTid INTEGER,text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER,end INTEGER,globalTid INTEGER,
                correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,end INTEGER,graphId INTEGER,
                graphNodeId INTEGER,correlationId INTEGER,
                demangledName INTEGER);
            INSERT INTO StringIds VALUES
                (1,'cudaGraphLaunch_v10000'),
                (2,'flash_attn_ext_f16<256>'),
                (3,'flash_attn_ext_vec<256,1,30,30>'),
                (4,'void convert_unary_cont_vec4<__nv_bfloat16, __half>(void)'),
                (5,'soft_max_f32<256>'),
                (6,'cudaLaunchKernel_v7000');
        """)
        idx=0
        for phase,num,graph_n,graph_id in (
            ("PREFILL",96,68,2),("DECODE",64,60,10)):
            for j in range(num):
                idx+=1
                t=idx*100000
                db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,7,?)",
                           (t,t+40000,"RUTH_GEMMA3_PHASE28_"+phase))
                if j < graph_n:
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                        "VALUES (?,?,7,?,1)",(t+10,t+30,idx))
                    names=(2,4) if phase=="PREFILL" else (3,)
                    for k,nom in enumerate(names):
                        db.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                            "VALUES (?,?,?,?,?,?)",
                            (t+1000+k*10,t+1005+k*10,graph_id,
                             graph_id*100+k,idx,nom))
                elif phase=="PREFILL" and j == graph_n:
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                        "VALUES (?,?,7,9999,6)",(t+10,t+30))
                    for name in (4,5):
                        db.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                            "VALUES (?,?,?,?,?,?)",
                            (t+2000,t+2010,None,None,9999,name))


class TestPhase28PrecisionTrace(unittest.TestCase):
    def test_identify_conversions_and_unknown_implementations(self):
        self.assertEqual(
            classify_conversion(
                "void convert_unary_cont_vec4<__nv_bfloat16, __half>(void)"),
            {"source_dtype":"BF16","target_dtype":"F16",
             "recognition":"source_target_template_verified"})
        self.assertEqual(
            classify_conversion("convert_unary<int16_t,float>"),
            {"source_dtype":"I16","target_dtype":"F32",
             "recognition":"source_target_template_verified"})
        self.assertIsNone(classify_conversion("matmul_fp32"))

    def test_same_capture_graph_mapping_and_direct_name_evidence(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"nodes.sqlite"
            write_nsight(fp)
            out=inspect_precision_trace(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertEqual(out["status"],
                             "precision_kernel_name_survey_observed")
            self.assertTrue(out["kernel_name_survey_verified"])
            self.assertTrue(out["graph_phase_causal_mapping_verified"])
            self.assertEqual(out["conversion_kernel_records"],69)
            self.assertEqual(len(out["conversion_summary"]),1)
            row=out["conversion_summary"][0]
            self.assertEqual(row["by_graph_id"],{"2":68})
            self.assertEqual(row["by_phase"],{"prefill":69})
            self.assertEqual(row["kernel_gpu_ms_sum"],0.00035)
            self.assertEqual(
                out["unattributed_direct_conversion_records"],0)
            self.assertEqual(
                out["unattributed_graph_conversion_records"],0)
            self.assertEqual(
                out["other_precision_name_evidence"][0]["operator_family"],
                "SOFTMAX_F32")
            self.assertFalse(out["source_precision_runtime_inventory_complete"])
            self.assertFalse(out["production_qualified"])


if __name__=="__main__":
    unittest.main()
