"""Read-only schema capability survey for an existing pinned Phase28 Nsight export.

This deliberately inspects SQLite table/column NAMES only, never operand
values, pointer values, CUDA API parameters or model data. Presence of a
candidate column is not evidence of actual available GPU buffer provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_dispatch import inspect_export
from .gemma3_phase28_gate import verify_plan

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
_MAX_TABLES = 512
_MAX_COLUMNS_PER_TABLE = 256
_MAX_CANDIDATES = 60
_REQUIRED = {
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "StringIds",
    "NVTX_EVENTS",
}
_EXACT_ACTIVITY = (
    "CUPTI_ACTIVITY_KIND_KERNEL",
    "CUPTI_ACTIVITY_KIND_RUNTIME",
    "NVTX_EVENTS",
)


def _has_dag_edges(table: str, columns: tuple[str,...]) -> bool:
    """A heuristic for an *explicit* graph dependency table, not a DAG proof."""
    name=table.lower()
    cols={c.lower() for c in columns}
    if "graph" in name and ("edge" in name or "depend" in name):
        return True
    pairings=(
        ("srcnodeid","dstnodeid"),
        ("sourcenodeid","targetnodeid"),
        ("fromnodeid","tonodeid"),
        ("parentnodeid","childnodeid"),
    )
    return "graph" in name and any({a,b} <= cols for a,b in pairings)


def inspect_nsight_schema(
    file: Path, *, candidate: str="bf16-on", n_ctx: int=2048,
    root: Path=PROJECT_ROOT,
) -> dict:
    verify_plan(candidate,n_ctx)
    if candidate != "bf16-on" or n_ctx != 2048:
        raise ValueError("Only pinned BF16 ON context 2048 is approved")
    base=inspect_export(file,root,expected_flash="on")
    report={
        "status":"nsight_schema_capability_unverified",
        "candidate":candidate,"n_ctx":n_ctx,
        "sqlite_tables_observed":0,
        "schema_survey_verified":False,
        "expected_flash_consistent_with_kernel_names":base[
            "expected_flash_consistent_with_kernel_names"],
        "required_activity_tables_present":False,
        "activity_column_names":{},
        "explicit_graph_edge_table_candidates":[],
        "pointer_or_buffer_column_candidates":[],
        "api_parameter_column_candidates":[],
        "structural_join_fields":{},
        "explicit_graph_edge_data_verified":False,
        "raw_operand_pointer_values_examined":False,
        "cublas_operand_pointer_identity_verified":False,
        "observed_cast_gemm_dependency_verified":False,
        "ggml_operator_identity_verified":False,
        "production_qualified":False,
        "gpu_inference_executed":False,
        "evidence_scope":"SQLite schema names only, not column values",
    }
    if base["expected_flash_consistent_with_kernel_names"] is not True:
        return report
    root=Path(root)
    path=((root/file) if not file.is_absolute() else file).resolve()
    with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as conn:
        raw=list(conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"))
        names=[name for (name,) in raw]
        report["sqlite_tables_observed"]=len(names)
        if len(names)>_MAX_TABLES or not _REQUIRED <= set(names):
            return report
        report["required_activity_tables_present"]=True
        all_cols={}
        for table in names:
            if not isinstance(table,str) or not _IDENTIFIER.fullmatch(table):
                return report
            cols=tuple(row[1] for row in conn.execute(
                'PRAGMA table_info("'+table+'")'))
            if (len(cols)>_MAX_COLUMNS_PER_TABLE or
                any(not isinstance(c,str) or
                    not _IDENTIFIER.fullmatch(c) for c in cols)):
                return report
            all_cols[table]=cols
        report["activity_column_names"]={
            k:list(all_cols[k]) for k in _EXACT_ACTIVITY
        }
        joins={}
        for name in _EXACT_ACTIVITY:
            cols=set(all_cols[name])
            joins[name]=sorted(cols & {
                "correlationId","graphId","graphNodeId","streamId",
                "globalTid","start","end","nameId","text","textId",
            })
        report["structural_join_fields"]=joins
        pointers=[]
        params=[]
        dag=[]
        for name,cols in all_cols.items():
            if _has_dag_edges(name,cols):
                dag.append({"table":name,"columns":list(cols)[:32]})
            for col in cols:
                low=col.lower()
                if any(token in low for token in (
                    "ptr","pointer","address","bufferoffset",
                    "devicepointer","deviceptr","hostptr")):
                    pointers.append({"table":name,"column":col})
                if any(token in low for token in (
                    "kernelparam","apiparam","argument","argvalue",
                    "params","parameter")):
                    params.append({"table":name,"column":col})
        if (len(dag)>_MAX_CANDIDATES or
            len(pointers)>_MAX_CANDIDATES or
            len(params)>_MAX_CANDIDATES):
            return report
        report["explicit_graph_edge_table_candidates"]=dag
        report["pointer_or_buffer_column_candidates"]=pointers
        report["api_parameter_column_candidates"]=params
        report["schema_survey_verified"]=True
        report["status"]="nsight_sqlite_schema_names_surveyed"
    return report


def main(argv: list[str] | None=None) -> int:
    p=argparse.ArgumentParser(
        description="Offline Phase28 Nsight schema capability survey (no values)")
    p.add_argument("--sqlite",type=Path,required=True)
    p.add_argument("--candidate",choices=("bf16-on",),required=True)
    p.add_argument("--n-ctx",type=int,required=True)
    args=p.parse_args(argv)
    try:
        result=inspect_nsight_schema(
            args.sqlite,candidate=args.candidate,n_ctx=args.n_ctx)
    except (ValueError,OSError,sqlite3.Error) as exc:
        result={
            "status":"nsight_schema_capability_unverified",
            "error":type(exc).__name__+": "+str(exc)[:200],
            "schema_survey_verified":False,
            "cublas_operand_pointer_identity_verified":False,
            "observed_cast_gemm_dependency_verified":False,
            "production_qualified":False,"gpu_inference_executed":False,
        }
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result["schema_survey_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())
