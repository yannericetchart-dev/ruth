"""Offline synthetic Nsight 2025.6 SQLite graph replay cases; no GPU."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_graph_audit import inspect_graph_replays


def make_trace(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER,
                correlationId INTEGER, demangledName INTEGER,
                graphNodeId INTEGER);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,end INTEGER,eventType INTEGER,
                globalTid INTEGER,text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER,end INTEGER,correlationId INTEGER,
                globalTid INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER (
                start INTEGER,end INTEGER,globalTid INTEGER,
                correlationId INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE (
                start INTEGER,end INTEGER,correlationId INTEGER,
                graphId INTEGER,graphExecId INTEGER);
            CREATE TABLE CUDA_GRAPH_EVENTS (start INTEGER,end INTEGER);
            INSERT INTO StringIds VALUES
              (1,'flash_attn_ext_vec<256>'),
              (2,'cudaGraphLaunch_v10000'),
              (3,'cuGraphLaunch');
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
              VALUES (1,2,1,1,101);
        """)
        cid=10
        for phase,count in (("PREFILL",8),("DECODE",64)):
            for _ in range(count):
                cid+=1
                t=cid*1000000
                db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,59,123,?)",
                           (t,t+100000,"RUTH_GEMMA3_PHASE28_"+phase))
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES "
                    "(?,?,?,?,2)",(t+100,t+200,cid,123))
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES "
                    "(?,?,?,?,3)",(t+110,t+190,123,cid))
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                    "VALUES (?,?,?,?,?)",
                    (t+300,t+400,cid,7,77))


class TestPhase28GraphAudit(unittest.TestCase):
    def test_fully_correlated_launches_are_not_exhaustive_kernel_coverage(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            base=root/".ruth-build/phase28"
            base.mkdir(parents=True)
            file=base/"graph.sqlite"
            make_trace(file)
            report=inspect_graph_replays(
                file,candidate="bf16-on",n_ctx=256,root=root)
            self.assertEqual(report["status"],"cuda_graph_replay_phase_correlated")
            self.assertEqual(report["runtime_cuda_graph_launches"],72)
            self.assertEqual(report["runtime_graph_launches_by_phase"],
                             {"prefill":8,"decode":64})
            self.assertEqual(report["graph_trace_rows"],72)
            self.assertEqual(report["graph_trace_distinct_graph_ids"],1)
            self.assertEqual(report["graph_trace_distinct_graph_exec_ids"],1)
            self.assertEqual(report["graph_trace_matches_driver_correlation"],72)
            self.assertEqual(report["graph_trace_matches_runtime_correlation"],0)
            self.assertEqual(report["graph_trace_launches_by_phase"],
                             {"prefill":8,"decode":64})
            self.assertTrue(report["graph_trace_to_phase_correlation_verified"])
            self.assertEqual(report["kernel_table_graph_columns"],["graphNodeId"])
            self.assertTrue(report["cuda_graph_phase_attribution_verified"])
            self.assertFalse(report["complete_attention_workload_coverage_verified"])
            self.assertFalse(report["graph_nodes_replayed_and_correlated_verified"])
            self.assertFalse(report["production_qualified"])

    def test_missing_or_unmatched_graph_events_fail_closed(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            base=root/".ruth-build/phase28"
            base.mkdir(parents=True)
            file=base/"graph.sqlite"
            make_trace(file)
            with sqlite3.connect(file) as db:
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
            bad=inspect_graph_replays(
                file,candidate="bf16-on",n_ctx=256,root=root)
            self.assertFalse(bad["cuda_graph_phase_attribution_verified"])
            with sqlite3.connect(file) as db:
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                           "VALUES (?,?,?,7,77)",(1,2,11))
                db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET "
                           "globalTid=999 WHERE correlationId=11")
            unmatched=inspect_graph_replays(
                file,candidate="bf16-on",n_ctx=256,root=root)
            self.assertFalse(unmatched["cuda_graph_phase_attribution_verified"])
            self.assertEqual(unmatched["unmatched_runtime_graph_launches"],1)

    def test_non_flash_trace_does_not_forge_flash_dispatch(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            base=root/".ruth-build/phase28"
            base.mkdir(parents=True)
            file=base/"graph.sqlite"
            make_trace(file)
            report=inspect_graph_replays(
                file,candidate="f16-off",n_ctx=256,root=root)
            self.assertFalse(report["expected_flash_consistent_with_kernel_names"])
            self.assertFalse(report["cuda_graph_phase_attribution_verified"])


if __name__=="__main__":
    unittest.main()
