"""Synthetic Phase28 CUDA Graph node-identifier coverage tests; no CUDA."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_aux_node_audit import inspect_aux_node_ids


def fixture(path: Path, *, with_nodes: bool=True,
            with_memcpy: bool=True, with_original: bool=True):
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO StringIds VALUES(1,'flash_attn_ext_f16<256>');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER,end INTEGER,demangledName INTEGER,
                graphNodeId INTEGER,graphId INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER,end INTEGER,globalTid INTEGER,
                correlationId INTEGER,nameId INTEGER);
            CREATE TABLE NVTX_EVENTS(
                start INTEGER,end INTEGER,globalTid INTEGER,
                text TEXT,eventType INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                VALUES(100,110,1,11,2,1001);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                VALUES(200,210,1,12,2,1002);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL
                VALUES(300,310,1,NULL,NULL,1003);
        """)
        if with_nodes:
            original=("originalGraphNodeId INTEGER" if with_original
                      else "unrelatedId INTEGER")
            db.execute("CREATE TABLE CUDA_GRAPH_NODE_EVENTS("
                       f"graphNodeId INTEGER,{original},"
                       "start INTEGER,end INTEGER,nameId INTEGER)")
            db.execute("INSERT INTO CUDA_GRAPH_NODE_EVENTS "
                       "VALUES(11,NULL,10,20,1)")
            db.execute("INSERT INTO CUDA_GRAPH_NODE_EVENTS "
                       "VALUES(12,11,10,20,1)")
            db.execute("INSERT INTO CUDA_GRAPH_NODE_EVENTS "
                       "VALUES(13,99,10,20,1)")
        if with_memcpy:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY("
                       "graphNodeId INTEGER,correlationId INTEGER,"
                       "virtualAddress TEXT,start INTEGER,end INTEGER)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY "
                       "VALUES(12,200,'SENSITIVE_PTR_VALUE',10,20)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY "
                       "VALUES(88,201,'SENSITIVE_PTR_VALUE',30,40)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY "
                       "VALUES(NULL,202,'SENSITIVE_PTR_VALUE',50,60)")


class TestPhase28AuxNodeAudit(unittest.TestCase):
    def inspect(self, **kwargs):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"capture.sqlite"
            fixture(fp,**kwargs)
            return inspect_aux_node_ids(
                fp,candidate="bf16-on",n_ctx=2048,root=root)

    def test_clone_lineage_counts_are_not_dag_edges(self):
        out=self.inspect()
        self.assertEqual(out["status"],
                         "auxiliary_graph_node_identifier_counts_observed")
        self.assertTrue(out["identifier_coverage_verified"])
        node=out["graph_node_creation_cloning_activity"]
        self.assertEqual(node["event_rows"],3)
        self.assertEqual(node["distinct_graph_node_ids"],3)
        self.assertEqual(node["with_original_graph_node_id"],2)
        self.assertEqual(node["with_same_original_node_id"],0)
        self.assertEqual(
            node["original_node_ids_found_among_graph_node_events"],1)
        lineage=out["kernel_node_original_id_mapping"]
        self.assertEqual(lineage["distinct_kernel_node_ids_checked"],2)
        self.assertEqual(lineage["no_original_node_candidate"],1)
        self.assertEqual(lineage["exactly_one_original_node_candidate"],1)
        self.assertEqual(lineage["multiple_original_node_candidates"],0)
        self.assertEqual(lineage["unique_original_node_found_in_events"],1)
        self.assertFalse(
            lineage["kernel_id_to_original_id_mapping_complete_unambiguous"])
        self.assertTrue(
            lineage["mapping_is_clone_lineage_not_dag_or_buffer_dependency"])
        kernels=out["kernel_graph_node_id_coverage"]
        self.assertEqual(kernels["distinct_nonzero_kernel_graph_node_ids"],2)
        self.assertEqual(
            kernels["distinct_kernel_node_ids_present_in_node_events"],2)
        self.assertEqual(
            kernels["distinct_kernel_node_ids_present_as_original_ids"],1)
        memcpy=out["memcpy_node_id_coverage"]
        self.assertTrue(memcpy["table_present"])
        self.assertEqual(memcpy["memcpy_rows"],3)
        self.assertEqual(memcpy["with_nonzero_graph_node_id"],2)
        self.assertEqual(memcpy["graph_node_ids_present_in_node_events"],1)
        self.assertEqual(memcpy["graph_node_ids_present_in_kernel_activity"],1)
        self.assertNotIn("SENSITIVE_PTR_VALUE",str(out))
        self.assertFalse(out["explicit_graph_dependency_edges_verified"])
        self.assertFalse(out["kernel_operand_buffer_identity_verified"])
        self.assertFalse(out["cast_to_gemm_producer_consumer_verified"])
        self.assertFalse(out["production_qualified"])

    def test_conflicting_original_ids_are_counted_as_ambiguous(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"capture.sqlite"
            fixture(fp)
            with sqlite3.connect(fp) as db:
                # Kernel node 12 now names two different candidate origins:
                # node 11 and node 13, both in node-event metadata.
                db.execute(
                    "INSERT INTO CUDA_GRAPH_NODE_EVENTS "
                    "VALUES(12,13,25,30,1)")
            out=inspect_aux_node_ids(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            lineage=out["kernel_node_original_id_mapping"]
            self.assertEqual(lineage["distinct_kernel_node_ids_checked"],2)
            self.assertEqual(lineage["no_original_node_candidate"],1)
            self.assertEqual(lineage["exactly_one_original_node_candidate"],0)
            self.assertEqual(lineage["multiple_original_node_candidates"],1)
            self.assertEqual(lineage["unique_original_node_found_in_events"],0)
            self.assertFalse(
                lineage["kernel_id_to_original_id_mapping_complete_unambiguous"])
            self.assertFalse(out["cast_to_gemm_producer_consumer_verified"])

    def test_unique_cloned_kernel_nodes_are_lineage_not_dataflow(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"capture.sqlite"
            fixture(fp)
            with sqlite3.connect(fp) as db:
                # Node 11 acquires a distinct, resolvable parent 13.
                db.execute(
                    "UPDATE CUDA_GRAPH_NODE_EVENTS "
                    "SET originalGraphNodeId=13 WHERE graphNodeId=11")
            out=inspect_aux_node_ids(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            lineage=out["kernel_node_original_id_mapping"]
            self.assertEqual(lineage["exactly_one_original_node_candidate"],2)
            self.assertEqual(lineage["unique_original_node_found_in_events"],2)
            self.assertTrue(
                lineage["kernel_id_to_original_id_mapping_complete_unambiguous"])
            self.assertFalse(out["explicit_graph_dependency_edges_verified"])
            self.assertFalse(out["cast_to_gemm_producer_consumer_verified"])
            self.assertFalse(out["kernel_operand_buffer_identity_verified"])

    def test_without_memcpy_still_counts_graph_creation(self):
        out=self.inspect(with_memcpy=False)
        self.assertTrue(out["identifier_coverage_verified"])
        self.assertEqual(out["memcpy_node_id_coverage"],{
            "table_present":False})
        self.assertEqual(
            out["graph_node_creation_cloning_activity"]["event_rows"],3)

    def test_graph_event_table_absence_fails_closed(self):
        out=self.inspect(with_nodes=False)
        self.assertTrue(out["schema_survey_verified"])
        self.assertFalse(out["identifier_coverage_verified"])
        self.assertEqual(out["status"],
                         "cuda_graph_node_creation_events_absent")

    def test_original_node_id_column_absence_fails_closed(self):
        out=self.inspect(with_original=False)
        self.assertFalse(out["identifier_coverage_verified"])
        self.assertEqual(out["status"],
                         "cuda_graph_node_identifiers_missing")


if __name__=="__main__":
    unittest.main()
