"""Synthetic read-only SQLite schema capability tests: no CUDA or model."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from ruth_runtime.gemma3_phase28_nsight_schema_audit import (
    _has_dag_edges, inspect_nsight_schema,
)


def sample_sqlite(path: Path, *, added_data: bool=False,
                  graph_edges: bool=False, runtime: bool=True):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY,value TEXT)")
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
            "start INTEGER,end INTEGER,demangledName INTEGER,"
            "correlationId INTEGER,graphId INTEGER,graphNodeId INTEGER,"
            "streamId INTEGER)")
        if added_data:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                       "ADD COLUMN devicePointerAddress TEXT")
        if runtime:
            db.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
                "correlationId INTEGER,globalTid INTEGER,"
                "start INTEGER,end INTEGER,nameId INTEGER)")
        if added_data and runtime:
            db.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
                       "ADD COLUMN kernelParameters TEXT")
        db.execute(
            "CREATE TABLE NVTX_EVENTS ("
            "eventType INTEGER,start INTEGER,end INTEGER,"
            "globalTid INTEGER,text TEXT)")
        if graph_edges:
            db.execute(
                "CREATE TABLE CUDA_GRAPH_DEPENDENCY_EDGES ("
                "graphId INTEGER,srcNodeId INTEGER,dstNodeId INTEGER)")
        db.execute(
            "INSERT INTO StringIds VALUES "
            "(1,'flash_attn_ext_f16<256>')")
        db.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(start,end,demangledName,correlationId,graphId,"
            "graphNodeId,streamId) VALUES (1000,2000,1,10,2,1,14)")
        if added_data:
            # Deliberately secret-looking synthetic values: the tool
            # must NOT inspect or output actual values.
            db.execute(
                "UPDATE CUPTI_ACTIVITY_KIND_KERNEL "
                "SET devicePointerAddress='SECRET_NOT_TO_EXPOSE'")
            if runtime:
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
                    "(kernelParameters) VALUES ('SECRET_PARAM_NOT_TO_EXPOSE')")


class TestNsightSchemaAudit(unittest.TestCase):
    def run_on_sample(self, **kwargs):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            file=folder/"synthetic.sqlite"
            sample_sqlite(file,**kwargs)
            return inspect_nsight_schema(
                file,candidate="bf16-on",n_ctx=2048,root=root)

    def test_ordinary_activity_metadata_is_not_buffer_dependency(self):
        report=self.run_on_sample()
        self.assertEqual(report["status"],
                         "nsight_sqlite_schema_names_surveyed")
        self.assertTrue(report["schema_survey_verified"])
        self.assertTrue(report["required_activity_tables_present"])
        self.assertEqual(report["sqlite_tables_observed"],4)
        self.assertEqual(report["explicit_graph_edge_table_candidates"],[])
        self.assertEqual(report["pointer_or_buffer_column_candidates"],[])
        self.assertEqual(report["api_parameter_column_candidates"],[])
        self.assertIn(
            "graphNodeId",
            report["structural_join_fields"]["CUPTI_ACTIVITY_KIND_KERNEL"])
        self.assertFalse(report["observed_cast_gemm_dependency_verified"])
        self.assertFalse(report["production_qualified"])
        self.assertFalse(report["gpu_inference_executed"])

    def test_candidate_columns_do_not_upgrade_into_dataflow_evidence(self):
        report=self.run_on_sample(added_data=True,graph_edges=True)
        self.assertTrue(report["schema_survey_verified"])
        self.assertEqual(len(report["explicit_graph_edge_table_candidates"]),1)
        self.assertEqual(report["pointer_or_buffer_column_candidates"],[
            {"table":"CUPTI_ACTIVITY_KIND_KERNEL",
             "column":"devicePointerAddress"}])
        self.assertEqual(report["api_parameter_column_candidates"],[
            {"table":"CUPTI_ACTIVITY_KIND_RUNTIME",
             "column":"kernelParameters"}])
        txt=str(report)
        self.assertNotIn("SECRET_NOT_TO_EXPOSE",txt)
        self.assertNotIn("SECRET_PARAM_NOT_TO_EXPOSE",txt)
        self.assertFalse(report["raw_operand_pointer_values_examined"])
        self.assertFalse(report["explicit_graph_edge_data_verified"])
        self.assertFalse(report["cublas_operand_pointer_identity_verified"])
        self.assertFalse(report["observed_cast_gemm_dependency_verified"])

    def test_auxiliary_counts_are_only_metadata_not_pointer_proof(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            fp=folder/"synthetic.sqlite"
            sample_sqlite(fp,added_data=True,graph_edges=True)
            with sqlite3.connect(fp) as db:
                db.execute(
                    "CREATE TABLE CUBLAS_EVENTS (start INTEGER,end INTEGER,"
                    "globalTid INTEGER,nameId INTEGER)")
                db.execute("INSERT INTO CUBLAS_EVENTS VALUES (1,2,3,4)")
                db.execute(
                    "CREATE TABLE CUDA_GRAPH_NODE_EVENTS ("
                    "start INTEGER,end INTEGER,graphNodeId INTEGER,"
                    "originalGraphNodeId INTEGER)")
                db.execute(
                    "INSERT INTO CUDA_GRAPH_NODE_EVENTS VALUES (1,2,10,9)")
                db.execute(
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY ("
                    "start INTEGER,end INTEGER,bytes INTEGER,"
                    "virtualAddress TEXT,graphNodeId INTEGER)")
                db.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY "
                    "VALUES (1,2,64,'SECRET_PAGE_ADDRESS',10)")
            out=inspect_nsight_schema(
                fp,candidate="bf16-on",n_ctx=2048,root=root)
            self.assertTrue(out["schema_survey_verified"])
            aux=out["auxiliary_event_table_metadata"]
            self.assertEqual(aux["CUBLAS_EVENTS"]["record_count"],1)
            self.assertEqual(
                aux["CUDA_GRAPH_NODE_EVENTS"]["record_count"],1)
            self.assertEqual(
                aux["CUPTI_ACTIVITY_KIND_MEMCPY"]["record_count"],1)
            self.assertFalse(aux["CUDA_GRAPH_EVENTS"]["present"])
            self.assertIn("originalGraphNodeId",
                          aux["CUDA_GRAPH_NODE_EVENTS"]["join_columns"])
            self.assertNotIn("SECRET_PAGE_ADDRESS",str(out))
            self.assertFalse(out["raw_operand_pointer_values_examined"])
            self.assertFalse(out["cublas_operand_pointer_identity_verified"])
            self.assertFalse(out["observed_cast_gemm_dependency_verified"])

    def test_missing_cuda_runtime_table_does_not_pass(self):
        report=self.run_on_sample(runtime=False)
        self.assertFalse(report["required_activity_tables_present"])
        self.assertFalse(report["schema_survey_verified"])

    def test_symlinked_sqlite_file_is_rejected_by_existing_gate(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/".ruth-build/phase28"
            folder.mkdir(parents=True)
            file=folder/"actual.sqlite"
            sample_sqlite(file)
            link=folder/"alias.sqlite"
            link.symlink_to(file)
            with self.assertRaises(ValueError):
                inspect_nsight_schema(
                    link,candidate="bf16-on",n_ctx=2048,root=root)

    def test_dag_detector_requires_graph_and_edges_not_plain_node_ids(self):
        self.assertFalse(_has_dag_edges(
            "CUPTI_ACTIVITY_KIND_KERNEL",
            ("graphId","graphNodeId","correlationId")))
        self.assertFalse(_has_dag_edges(
            "OTHER_TABLE",("srcNodeId","dstNodeId")))
        self.assertTrue(_has_dag_edges(
            "CUDA_GRAPH_EDGE",("graphId",)))
        self.assertTrue(_has_dag_edges(
            "CUDA_GRAPH_STRUCTURE",
            ("graphId","srcNodeId","dstNodeId")))


if __name__=="__main__":
    unittest.main()
