"""Offline, read-only identifier-coverage audit for Phase28 CUDA Graph metadata.

CUDA_GRAPH_NODE_EVENTS records node creation/cloning activity. An observed
originalGraphNodeId is a lineage hint, not a CUDA Graph dependency edge.
No raw pointers, copy payloads, NVTX contents, or API argument values are read.
No GPU allocations or inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_nsight_schema_audit import inspect_nsight_schema


def _scalar(db: sqlite3.Connection, sql: str) -> int:
    value=db.execute(sql).fetchone()[0]
    if not isinstance(value,int) or value<0:
        raise ValueError("Invalid aggregate count")
    return value


def inspect_aux_node_ids(
    file: Path, *, candidate: str="bf16-on", n_ctx: int=2048,
    root: Path=PROJECT_ROOT,
) -> dict:
    gate=inspect_nsight_schema(
        file,candidate=candidate,n_ctx=n_ctx,root=root)
    result={
        "status":"auxiliary_node_identifiers_unverified",
        "candidate":candidate,"n_ctx":n_ctx,
        "schema_survey_verified":gate["schema_survey_verified"],
        "identifier_coverage_verified":False,
        "graph_node_creation_cloning_activity":{},
        "kernel_graph_node_id_coverage":{},
        "kernel_node_original_id_mapping":{},
        "memcpy_node_id_coverage":{},
        "explicit_graph_dependency_edges_verified":False,
        "kernel_operand_buffer_identity_verified":False,
        "cast_to_gemm_producer_consumer_verified":False,
        "ggml_operator_identity_verified":False,
        "production_qualified":False,
        "gpu_inference_executed":False,
        "evidence_scope":"Aggregate counts of graph node IDs, no raw values",
    }
    if gate["schema_survey_verified"] is not True:
        return result
    aux=gate["auxiliary_event_table_metadata"]
    if not aux["CUDA_GRAPH_NODE_EVENTS"]["present"]:
        result["status"]="cuda_graph_node_creation_events_absent"
        return result
    need={"graphNodeId","originalGraphNodeId"}
    observed=set(aux["CUDA_GRAPH_NODE_EVENTS"]["join_columns"])
    if not need <= observed:
        result["status"]="cuda_graph_node_identifiers_missing"
        return result
    kernel_cols=set(gate["activity_column_names"][
        "CUPTI_ACTIVITY_KIND_KERNEL"])
    if "graphNodeId" not in kernel_cols:
        return result
    root=Path(root)
    p=((root/file) if not file.is_absolute() else file).resolve()
    with sqlite3.connect(p.as_uri()+"?mode=ro",uri=True) as db:
        n="CUDA_GRAPH_NODE_EVENTS"
        k="CUPTI_ACTIVITY_KIND_KERNEL"
        m="CUPTI_ACTIVITY_KIND_MEMCPY"
        # Counts contain no kernel arguments or memory addresses. 
        node={
            "event_rows":_scalar(db,f'SELECT COUNT(*) FROM "{n}"'),
            "distinct_graph_node_ids":_scalar(
                db,f'SELECT COUNT(DISTINCT graphNodeId) FROM "{n}" '
                   'WHERE graphNodeId IS NOT NULL AND graphNodeId!=0'),
            "with_original_graph_node_id":_scalar(
                db,f'SELECT COUNT(*) FROM "{n}" '
                   'WHERE originalGraphNodeId IS NOT NULL '
                   'AND originalGraphNodeId!=0'),
            "with_same_original_node_id":_scalar(
                db,f'SELECT COUNT(*) FROM "{n}" '
                   'WHERE originalGraphNodeId IS NOT NULL '
                   'AND originalGraphNodeId=graphNodeId'),
            "original_node_ids_found_among_graph_node_events":_scalar(
                db,f'SELECT COUNT(*) FROM "{n}" e '
                   'WHERE e.originalGraphNodeId IS NOT NULL '
                   'AND e.originalGraphNodeId!=0 AND EXISTS '
                   f'(SELECT 1 FROM "{n}" o '
                   'WHERE o.graphNodeId=e.originalGraphNodeId)'),
        }
        # This is identifier equality between metadata records and
        # kernel activity records, NOT a graph dependency edge.
        kernel={
            "distinct_nonzero_kernel_graph_node_ids":_scalar(
                db,f'SELECT COUNT(DISTINCT graphNodeId) FROM "{k}" '
                   'WHERE graphNodeId IS NOT NULL AND graphNodeId!=0'),
            "distinct_kernel_node_ids_present_in_node_events":_scalar(
                db,f'SELECT COUNT(DISTINCT k.graphNodeId) FROM "{k}" k '
                   'WHERE k.graphNodeId IS NOT NULL AND k.graphNodeId!=0 '
                   f'AND EXISTS (SELECT 1 FROM "{n}" e '
                   'WHERE e.graphNodeId=k.graphNodeId)'),
            "distinct_kernel_node_ids_present_as_original_ids":_scalar(
                db,f'SELECT COUNT(DISTINCT k.graphNodeId) FROM "{k}" k '
                   'WHERE k.graphNodeId IS NOT NULL AND k.graphNodeId!=0 '
                   f'AND EXISTS (SELECT 1 FROM "{n}" e '
                   'WHERE e.originalGraphNodeId=k.graphNodeId)'),
        }
        # Aggregate by distinct observed kernel IDs. A source node in
        # originalGraphNodeId denotes cloning lineage, never a DAG edge.
        # No raw node identifiers are returned to Python or reported.
        lineage_sql=f"""
            WITH kernel_ids AS (
                SELECT DISTINCT graphNodeId AS id FROM "{k}"
                WHERE graphNodeId IS NOT NULL AND graphNodeId!=0
            ), mapped AS (
                SELECT ki.id,
                    COUNT(DISTINCT CASE
                        WHEN e.originalGraphNodeId IS NOT NULL
                         AND e.originalGraphNodeId!=0
                        THEN e.originalGraphNodeId END) AS origins,
                    COUNT(DISTINCT CASE
                        WHEN e.originalGraphNodeId IS NOT NULL
                         AND e.originalGraphNodeId!=0 AND EXISTS (
                            SELECT 1 FROM "{n}" parent
                            WHERE parent.graphNodeId=e.originalGraphNodeId)
                        THEN e.originalGraphNodeId END) AS resolved
                FROM kernel_ids ki
                LEFT JOIN "{n}" e ON e.graphNodeId=ki.id
                GROUP BY ki.id
            )
            SELECT COUNT(*),
                   COALESCE(SUM(origins=0),0),
                   COALESCE(SUM(origins=1),0),
                   COALESCE(SUM(origins>1),0),
                   COALESCE(SUM(origins=1 AND resolved=1),0)
            FROM mapped
        """
        total,none,single,multiple,resolved=db.execute(
            lineage_sql).fetchone()
        if (any(type(v) is not int or v<0 for v in
                (total,none,single,multiple,resolved))
            or total!=none+single+multiple or resolved>single
            or total!=kernel["distinct_nonzero_kernel_graph_node_ids"]):
            return result
        lineage={
            "distinct_kernel_node_ids_checked":total,
            "no_original_node_candidate":none,
            "exactly_one_original_node_candidate":single,
            "multiple_original_node_candidates":multiple,
            "unique_original_node_found_in_events":resolved,
            "kernel_id_to_original_id_mapping_complete_unambiguous":bool(
                total>0 and single==resolved==total),
            "mapping_is_clone_lineage_not_dag_or_buffer_dependency":True,
        }
        mem={"table_present":aux[m]["present"]}
        if aux[m]["present"]:
            columns=set(aux[m]["join_columns"])
            if "graphNodeId" in columns:
                mem["memcpy_rows"]= _scalar(db,
                    f'SELECT COUNT(*) FROM "{m}"')
                mem["with_nonzero_graph_node_id"]=_scalar(
                    db,f'SELECT COUNT(*) FROM "{m}" '
                       'WHERE graphNodeId IS NOT NULL AND graphNodeId!=0')
                mem["graph_node_ids_present_in_node_events"]=_scalar(
                    db,f'SELECT COUNT(*) FROM "{m}" mc '
                       'WHERE mc.graphNodeId IS NOT NULL '
                       'AND mc.graphNodeId!=0 AND EXISTS '
                       f'(SELECT 1 FROM "{n}" e '
                       'WHERE e.graphNodeId=mc.graphNodeId)')
                mem["graph_node_ids_present_in_kernel_activity"]=_scalar(
                    db,f'SELECT COUNT(*) FROM "{m}" mc '
                       'WHERE mc.graphNodeId IS NOT NULL '
                       'AND mc.graphNodeId!=0 AND EXISTS '
                       f'(SELECT 1 FROM "{k}" kk '
                       'WHERE kk.graphNodeId=mc.graphNodeId)')
            else:
                mem["metadata_graph_node_id_absent"]=True
        result["graph_node_creation_cloning_activity"]=node
        result["kernel_graph_node_id_coverage"]=kernel
        result["kernel_node_original_id_mapping"]=lineage
        result["memcpy_node_id_coverage"]=mem
        result["identifier_coverage_verified"]=True
        result["status"]="auxiliary_graph_node_identifier_counts_observed"
    return result


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser(
        description="Read-only node creation/cloning and activity ID overlaps")
    p.add_argument("--sqlite",type=Path,required=True)
    p.add_argument("--candidate",choices=("bf16-on",),required=True)
    p.add_argument("--n-ctx",type=int,required=True)
    args=p.parse_args(argv)
    try:
        result=inspect_aux_node_ids(
            args.sqlite,candidate=args.candidate,n_ctx=args.n_ctx)
    except (OSError,sqlite3.Error,ValueError) as exc:
        result={
            "status":"auxiliary_node_identifiers_unverified",
            "error":type(exc).__name__+": "+str(exc)[:180],
            "identifier_coverage_verified":False,
            "cast_to_gemm_producer_consumer_verified":False,
            "kernel_operand_buffer_identity_verified":False,
            "production_qualified":False,
            "gpu_inference_executed":False,
        }
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result.get("identifier_coverage_verified") is True else 2


if __name__=="__main__":
    raise SystemExit(main())
