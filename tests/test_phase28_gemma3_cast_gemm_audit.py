"""Synthetic Phase28 per-range cast/GEMM vs Graph replay audit tests.

Test data mimics 96 prefill + 64 decode ranges (4 benchmark runs)
and uses a stable set of CUDA Graph kernel nodes per replay.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_cast_gemm_audit import (
    audit_cast_gemm, identify_target)


def make_trace(path: Path):
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
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
                (2,'cudaLaunchKernel_v7000'),
                (3,'void convert_unary_cont_vec4<float,__nv_bfloat16>(...)'),
                (4,'ampere_s16816gemm_bf16_128x64_ldg8_stages_32x6_tn'),
                (5,'void convert_unary_cont_vec4<__nv_bfloat16,__half>(...)'),
                (6,'flash_attn_ext_f16<256>'),
                (7,'flash_attn_ext_vec<256,1,30,30>');
        """)
        launch_cid=100
        direct_cid=100000
        for phase,nranges,ngraph,graphid in (
            ("PREFILL",96,68,2),("DECODE",64,60,10)):
            for j in range(nranges):
                outer=j if phase=="PREFILL" else j+96
                start=(outer+1)*1000000
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,59,7,?)",
                    (start,start+900000,
                     "RUTH_GEMMA3_PHASE28_"+phase))
                use_graph=j<ngraph
                if use_graph:
                    launch_cid+=1
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                        "VALUES (?,?,7,?,1)",
                        (start+10,start+30,launch_cid))
                    cid=launch_cid
                else:
                    cid=None
                operations=(
                    [(3,)] + [(4,)] + [(5,)]*52 + [(6,)]
                    if phase=="PREFILL" else [(7,)])
                for index,(name_id,) in enumerate(operations):
                    tick=start+1000+index*100
                    if use_graph:
                        node=graphid*1000+index
                        kernel_cid=cid
                        gid=graphid
                    else:
                        direct_cid+=1
                        kernel_cid=direct_cid
                        gid=None
                        node=None
                        db.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                            "VALUES (?,?,7,?,2)",
                            (start+100+index*10,
                             start+105+index*10,kernel_cid))
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                        "VALUES (?,?,?,?,?,?)",
                        (tick,tick+50,gid,node,kernel_cid,name_id))


class TestCastGemmAudit(unittest.TestCase):
    def test_only_exact_name_classes(self):
        self.assertEqual(identify_target(
            "convert_unary_cont_vec4<float,__nv_bfloat16>"),
            "F32_TO_BF16")
        self.assertEqual(identify_target(
            "convert_unary_cont_vec4<__nv_bfloat16,__half>"),
            "BF16_TO_F16")
        self.assertEqual(identify_target(
            "ampere_s16816gemm_bf16_128x64"),"BF16_GEMM")
        self.assertIsNone(identify_target("flash_attn_ext_f16<256>"))

    def test_fully_correlated_ranges_are_not_promoted_to_dataflow(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            dest=root/".ruth-build/phase28"
            dest.mkdir(parents=True)
            fp=dest/"node.sqlite"
            make_trace(fp)
            out=audit_cast_gemm(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertEqual(out["status"],
                             "cast_and_gemm_per_nvtx_range_accounted")
            self.assertTrue(out["per_range_attribution_verified"])
            self.assertTrue(out["prefill_per_range_count_balance_verified"])
            self.assertTrue(out["gpu_timeline_candidate_pairing_verified"])
            self.assertTrue(out["graph_node_kernel_order_stable"])
            self.assertEqual(out["prefill_casts_per_range_empirical"],1)
            self.assertEqual(out["overall_target_kernel_counts"],{
                "F32_TO_BF16":96,"BF16_GEMM":96,"BF16_TO_F16":4992})
            self.assertEqual(
                out["per_phase_path"]["prefill_graph_node"]["ranges"],68)
            self.assertEqual(
                out["per_phase_path"]["prefill_direct"]["ranges"],28)
            self.assertEqual(
                out["per_phase_path"]["decode_graph_node"]["ranges"],60)
            self.assertFalse(out["producer_consumer_buffer_dependency_verified"])
            self.assertFalse(out["ggml_operator_identity_verified"])
            self.assertFalse(out["production_qualified"])

    def test_equal_counts_but_wrong_gpu_time_order_is_not_pair_verified(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            dest=root/".ruth-build/phase28"
            dest.mkdir(parents=True)
            fp=dest/"node.sqlite"
            make_trace(fp)
            with sqlite3.connect(fp) as db:
                # Break GPU-time order for one direct prefill range,
                # without altering per-range target counts.
                db.execute(
                    "UPDATE CUPTI_ACTIVITY_KIND_KERNEL "
                    "SET start=start+500,end=end+500 "
                    "WHERE rowid=("
                    "SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "JOIN StringIds s ON k.demangledName=s.id "
                    "WHERE s.value LIKE 'void convert_unary_cont_vec4<float%' "
                    "AND k.graphId IS NULL LIMIT 1)")
            out=audit_cast_gemm(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertTrue(out["per_range_attribution_verified"])
            self.assertTrue(out["prefill_per_range_count_balance_verified"])
            self.assertFalse(out["gpu_timeline_candidate_pairing_verified"])
            self.assertFalse(out["producer_consumer_buffer_dependency_verified"])

    def test_missing_direct_cast_fails_closed(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            dest=root/".ruth-build/phase28"
            dest.mkdir(parents=True)
            fp=dest/"node.sqlite"
            make_trace(fp)
            with sqlite3.connect(fp) as db:
                db.execute(
                    "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "WHERE rowid=("
                    "SELECT k.rowid FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "JOIN StringIds s ON k.demangledName=s.id "
                    "WHERE s.value LIKE 'void convert_unary_cont_vec4<float%' "
                    "AND k.graphId IS NULL LIMIT 1)")
            out=audit_cast_gemm(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertFalse(out["per_range_attribution_verified"])
            self.assertFalse(out["prefill_per_range_count_balance_verified"])
            self.assertFalse(out["producer_consumer_buffer_dependency_verified"])


if __name__=="__main__":
    unittest.main()
