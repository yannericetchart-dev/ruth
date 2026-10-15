"""Synthetic offline tests for Phase28 per-NVTX direct-vs-graph dispatch."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_execution_paths import audit_execution_paths


def build_trace(file: Path) -> None:
    with sqlite3.connect(file) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,end INTEGER,eventType INTEGER,
                globalTid INTEGER,text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER,end INTEGER,correlationId INTEGER,
                globalTid INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,end INTEGER,correlationId INTEGER,
                demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
                start INTEGER,end INTEGER,correlationId INTEGER,
                graphId INTEGER,graphExecId INTEGER);
            INSERT INTO StringIds VALUES
                (0,'cudaLaunchKernel_v7000'),
                (1,'cudaGraphLaunch_v10000'),
                (2,'flash_attn_ext_f16<256>'),
                (3,'flash_attn_stream_k_fixup_uniform<256>'),
                (4,'flash_attn_ext_vec<256,1>'),
                (5,'flash_attn_combine_results<256>');
        """)
        cid=0
        for phase,nranges,ngraphs in (
            ("PREFILL",96,68),("DECODE",64,60)):
            for j in range(nranges):
                base=(j+1+ (0 if phase=="PREFILL" else 96))*100000
                db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,7,?)",
                           (base,base+85000,"RUTH_GEMMA3_PHASE28_"+phase))
                if j < ngraphs:
                    cid += 1
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                               "VALUES (?,?,?,?,1)",
                               (base+100,base+200,cid,7))
                    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                               "VALUES (?,?,?,?,?)",
                               (base+10000,base+12000,cid,
                                1 if phase=="PREFILL" else 2,
                                1 if phase=="PREFILL" else 2))
                else:
                    for layer in range(26):
                        for kname in ((2,3) if phase=="PREFILL" else (4,5)):
                            cid += 1
                            tick=base+300+100*layer+10*kname
                            db.execute(
                                "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                                "VALUES (?,?,?,?,0)",
                                (tick,tick+10,cid,7))
                            db.execute(
                                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                                "VALUES (?,?,?,?)",
                                (base+90000,base+90100,cid,kname))


class TestPhase28ExecutionPaths(unittest.TestCase):
    def test_68_and_60_graphs_plus_26_layers_for_remaining_ranges(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"trace.sqlite"
            build_trace(fp)
            report=audit_execution_paths(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertTrue(report["cuda_graph_phase_attribution_verified"])
            self.assertTrue(report["execution_path_classification_verified"])
            self.assertEqual(report["per_phase_execution_paths"],{
                "prefill":{"graph_replay":68,"direct_attention":28},
                "decode":{"graph_replay":60,"direct_attention":4},
            })
            self.assertEqual(report["direct_main_kernel_records_correlated"],832)
            self.assertEqual(report["direct_auxiliary_kernel_records_correlated"],832)
            self.assertEqual(report["graph_trace_launches_correlated"],128)
            self.assertEqual(report["ranges_without_dispatch_evidence"],0)
            self.assertFalse(report["cuda_graph_node_coverage_verified"])
            self.assertFalse(report["complete_attention_workload_coverage_verified"])
            self.assertFalse(report["production_qualified"])

    def test_missing_direct_kernel_is_rejected_without_claiming_completion(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"trace.sqlite"
            build_trace(fp)
            with sqlite3.connect(fp) as db:
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL "
                           "WHERE rowid=(SELECT MIN(rowid) FROM "
                           "CUPTI_ACTIVITY_KIND_KERNEL)")
            report=audit_execution_paths(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertFalse(report["execution_path_classification_verified"])
            self.assertEqual(report["ranges_with_unexpected_direct_kernel_counts"],1)

    def test_unmatched_direct_cuda_launch_is_rejected(self):
        with TemporaryDirectory() as temp:
            root=Path(temp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"trace.sqlite"
            build_trace(fp)
            with sqlite3.connect(fp) as db:
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                           "WHERE correlationId=(SELECT correlationId "
                           "FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1)")
            report=audit_execution_paths(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertFalse(report["execution_path_classification_verified"])
            self.assertEqual(report["direct_kernels_unattributed_or_ambiguous"],1)


if __name__=="__main__":
    unittest.main()
