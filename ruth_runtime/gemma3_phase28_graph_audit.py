"""Read-only Nsight CUDA Graph replay audit for Phase28.

This audit intentionally stops short of claiming complete kernel-node coverage.
CUPTI GRAPH_TRACE may record a graph *execution*, without tracing its nodes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_attribution import MARKERS
from .gemma3_phase28_dispatch import inspect_export
from .gemma3_phase28_gate import REPEATS, WARMUPS, verify_plan


def inspect_graph_replays(
    file: Path, *, candidate: str, n_ctx: int,
    root: Path = PROJECT_ROOT,
) -> dict:
    """Audit the exact named Phase28 trace, never infer unseen graph nodes."""
    plan = verify_plan(candidate, n_ctx)
    base_report = inspect_export(file, root, expected_flash=plan.flash_attention)
    output = {
        "status": "cuda_graph_replay_audit_unverified",
        "candidate": candidate,
        "n_ctx": n_ctx,
        "flash_attention": plan.flash_attention,
        "expected_flash_consistent_with_kernel_names":
            base_report["expected_flash_consistent_with_kernel_names"],
        "nvtx_phase_ranges": {},
        "expected_nvtx_phase_ranges": {
            "prefill": (WARMUPS + REPEATS) *
                ((plan.prompt_tokens + plan.n_batch - 1)//plan.n_batch),
            "decode": (WARMUPS + REPEATS)*plan.decode_tokens,
        },
        "cuda_graph_api_calls": {},
        "runtime_cuda_graph_launches": 0,
        "runtime_graph_launches_by_phase": {},
        "unmatched_runtime_graph_launches": 0,
        "ambiguous_runtime_graph_launches": 0,
        "graph_trace_rows": 0,
        "graph_trace_schema_columns": [],
        "graph_trace_distinct_graph_ids": None,
        "graph_trace_distinct_graph_exec_ids": None,
        "graph_trace_matches_runtime_correlation": 0,
        "graph_trace_matches_driver_correlation": 0,
        "kernel_table_graph_columns": [],
        "cuda_graph_phase_attribution_verified": False,
        "complete_attention_workload_coverage_verified": False,
        "graph_nodes_replayed_and_correlated_verified": False,
        "production_qualified": False,
    }
    path = (root / file) if not file.is_absolute() else file
    path = path.resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as conn:
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        def columns(table: str) -> set[str]:
            return {r[1] for r in conn.execute(
                "PRAGMA table_info(" + table + ")")} if table in names else set()
        nvc = columns("NVTX_EVENTS")
        runtime_c = columns("CUPTI_ACTIVITY_KIND_RUNTIME")
        driver_c = columns("CUPTI_ACTIVITY_KIND_DRIVER")
        graph_c = columns("CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        kernel_c = columns("CUPTI_ACTIVITY_KIND_KERNEL")
        output["graph_trace_schema_columns"] = sorted(graph_c)
        output["kernel_table_graph_columns"] = sorted(
            c for c in kernel_c if "graph" in c.lower())
        if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in names:
            output["graph_trace_rows"] = conn.execute(
                "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
            ).fetchone()[0]
            for field, outkey in (
                ("graphId", "graph_trace_distinct_graph_ids"),
                ("graphExecId", "graph_trace_distinct_graph_exec_ids"),
            ):
                if field in graph_c:
                    output[outkey] = conn.execute(
                        "SELECT COUNT(DISTINCT " + field + ") "
                        "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"
                    ).fetchone()[0]

        if not {"NVTX_EVENTS", "StringIds",
                "CUPTI_ACTIVITY_KIND_RUNTIME"} <= names:
            return output
        if not {"start","end","eventType","globalTid"} <= nvc:
            return output
        if not ({"start","end","globalTid","correlationId",
                 "nameId"} <= runtime_c):
            return output
        if not ("text" in nvc or "textId" in nvc):
            return output
        label = (
            "COALESCE(n.text,s.value)" if {"text","textId"} <= nvc
            else "n.text" if "text" in nvc else "s.value"
        )
        joined = (" LEFT JOIN StringIds s ON n.textId=s.id"
                  if "textId" in nvc else "")
        nvtx_sql = (
            "SELECT n.start,n.end,n.globalTid," + label +
            " FROM NVTX_EVENTS n" + joined + " WHERE n.eventType=59")
        ranges = defaultdict(list)
        nv_counts = Counter()
        for start,end,tid,name in conn.execute(nvtx_sql):
            if name not in MARKERS.values():
                continue
            if (type(start) is not int or type(end) is not int
                or type(tid) is not int or end <= start):
                return output
            phase = next(p for p,v in MARKERS.items() if v == name)
            ranges[tid].append((start,end,phase))
            nv_counts[phase] += 1
        output["nvtx_phase_ranges"] = dict(nv_counts)

        api_rows = conn.execute(
            "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId "
            "WHERE LOWER(s.value) LIKE '%graph%'").fetchall()
        output["cuda_graph_api_calls"] = dict(Counter(
            name for _,_,_,_,name in api_rows))
        launches = [r for r in api_rows if "cudagraphlaunch" in r[4].lower()]
        output["runtime_cuda_graph_launches"] = len(launches)
        by_phase = Counter()
        unmatched = 0
        ambiguous = 0
        runtime_ids = {}
        for start,end,tid,cid,name in launches:
            matches = set()
            if (type(start) is int and type(end) is int
                and type(tid) is int and start <= end):
                matches = {
                    p for t0,t1,p in ranges.get(tid,())
                    if t0 <= start and end <= t1
                }
            if len(matches) == 1:
                by_phase[next(iter(matches))] += 1
            elif not matches:
                unmatched += 1
            else:
                ambiguous += 1
            if type(cid) is int:
                runtime_ids[cid] = runtime_ids.get(cid,0) + 1
        output["runtime_graph_launches_by_phase"] = dict(by_phase)
        output["unmatched_runtime_graph_launches"] = unmatched
        output["ambiguous_runtime_graph_launches"] = ambiguous

        if ("correlationId" in graph_c
            and "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in names):
            trace_ids = [
                r[0] for r in conn.execute(
                    "SELECT correlationId "
                    "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
            ]
            output["graph_trace_matches_runtime_correlation"] = sum(
                type(cid) is int and runtime_ids.get(cid,0) == 1
                for cid in trace_ids)
            if {"correlationId","nameId"} <= driver_c:
                drv = Counter(r[0] for r in conn.execute(
                    "SELECT d.correlationId FROM CUPTI_ACTIVITY_KIND_DRIVER d "
                    "JOIN StringIds s ON s.id=d.nameId "
                    "WHERE LOWER(s.value) LIKE '%graphlaunch%'"))
                output["graph_trace_matches_driver_correlation"] = sum(
                    type(cid) is int and drv[cid] == 1
                    for cid in trace_ids)

        success = (
            base_report["expected_flash_consistent_with_kernel_names"] is True
            and dict(nv_counts) == output["expected_nvtx_phase_ranges"]
            and bool(launches) and not unmatched and not ambiguous
            and sum(by_phase.values()) == len(launches)
            and by_phase["prefill"] > 0 and by_phase["decode"] > 0
            and output["graph_trace_rows"] > 0
        )
        output["cuda_graph_phase_attribution_verified"] = success
        output["status"] = (
            "cuda_graph_replay_phase_correlated"
            if success else "cuda_graph_replay_audit_unverified"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only Phase28 CUDA Graph API/replay/NVTX audit")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--candidate", choices=(
        "f16-off","bf16-on","f32-off","mixed-on"), required=True)
    p.add_argument("--n-ctx", required=True, type=int)
    args = p.parse_args(argv)
    try:
        result = inspect_graph_replays(
            Path(args.sqlite), candidate=args.candidate, n_ctx=args.n_ctx)
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        result = {
            "status":"cuda_graph_replay_audit_unverified",
            "error":type(exc).__name__+": "+str(exc)[:200],
            "cuda_graph_phase_attribution_verified":False,
            "complete_attention_workload_coverage_verified":False,
            "production_qualified":False,
        }
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result["cuda_graph_phase_attribution_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
