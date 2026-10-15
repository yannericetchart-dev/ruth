"""Offline synthetic tests for Phase28 graph-node replay bookkeeping."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_node_audit import inspect_node_trace


def make_node_trace(file: Path):
    with sqlite3.connect(file) as db:
        db.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER,end INTEGER,eventType INTEGER,
                globalTid INTEGER,text TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
                start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER,end INTEGER,graphId INTEGER,
                graphNodeId INTEGER,demangledName INTEGER);
            INSERT INTO StringIds VALUES
                (1,'cudaGraphLaunch_v10000'),
                (2,'flash_attn_ext_f16<256>'),
                (3,'flash_attn_ext_vec<256,1,30,30>');
        """)
        ix = 0
        for phase,total,graph_launches,graph_id,number_nodes,kname in (
            ("PREFILL",96,68,2,3,2),
            ("DECODE",64,60,10,2,3)):
            for j in range(total):
                ix += 1
                start=ix*100000
                db.execute(
                    "INSERT INTO NVTX_EVENTS VALUES (?,?,59,7,?)",
                    (start,start+90000,
                     "RUTH_GEMMA3_PHASE28_"+phase))
                if j < graph_launches:
                    db.execute(
                        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                        "VALUES (?,?,7,1)",(start+100,start+200))
                    for node in range(number_nodes):
                        db.execute(
                            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                            "VALUES (?,?,?,?,?)",
                            (start+5000+node*20,start+5010+node*20,
                             graph_id,graph_id*100+node,kname))


class TestPhase28NodeAudit(unittest.TestCase):
    def test_graph_kernel_nodes_have_uniform_experimental_replay_counts(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"nodes.sqlite"
            make_node_trace(fp)
            out=inspect_node_trace(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertTrue(
                out["observed_graph_kernel_node_repetitions_verified"])
            self.assertEqual(out["runtime_graph_launches_by_phase"],
                             {"prefill":68,"decode":60})
            self.assertEqual(out["kernel_records_with_graph_and_node_id"],324)
            summary={r["graph_id"]:r for r in out["graph_node_kernel_summary"]}
            self.assertEqual(summary[2]["unique_kernel_node_ids"],3)
            self.assertEqual(summary[2]["node_launch_frequency_histogram"],
                             {"68":3})
            self.assertEqual(summary[10]["node_launch_frequency_histogram"],
                             {"60":2})
            self.assertEqual(out["graph_phase_candidates_by_count"],[
                {"graph_id":2,"candidate_nvtx_phases_by_count_only":["prefill"],
                 "repeat_count":68},
                {"graph_id":10,"candidate_nvtx_phases_by_count_only":["decode"],
                 "repeat_count":60},
            ])
            self.assertFalse(out["graph_to_nvtx_phase_verified"])
            self.assertFalse(out["all_cuda_graph_node_types_traced_verified"])
            self.assertFalse(out["complete_attention_workload_coverage_verified"])
            self.assertFalse(out["production_qualified"])

    def test_missing_kernel_node_execution_fails_closed(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"nodes.sqlite"
            make_node_trace(fp)
            with sqlite3.connect(fp) as db:
                db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL "
                           "WHERE rowid=(SELECT MIN(rowid) "
                           "FROM CUPTI_ACTIVITY_KIND_KERNEL)")
            out=inspect_node_trace(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertFalse(
                out["observed_graph_kernel_node_repetitions_verified"])
            self.assertFalse(
                out["graph_node_kernel_summary"][0][
                    "every_observed_kernel_node_repeated_equally"])

    def test_missing_nvtx_launch_and_partial_graph_id_fail_closed(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"nodes.sqlite"
            make_node_trace(fp)
            with sqlite3.connect(fp) as db:
                db.execute(
                    "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=NULL "
                    "WHERE rowid=1")
            out=inspect_node_trace(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertEqual(out["kernel_records_with_partial_graph_identity"],1)
            self.assertFalse(
                out["observed_graph_kernel_node_repetitions_verified"])


if __name__=="__main__":
    unittest.main()
