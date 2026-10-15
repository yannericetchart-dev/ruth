"""Offline per-NVTX-range CUDA Graph versus directly launched attention audit.

No CUDA inference and no external profiler launch. This verifies host launch
paths; it never asserts that CUDA Graph node executions are individually traced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_attribution import MARKERS, MAIN_FLASH
from .gemma3_phase28_dispatch import classify_kernel, inspect_export
from .gemma3_phase28_gate import REPEATS, WARMUPS, verify_plan
from .gemma3_phase28_graph_audit import inspect_graph_replays

# Gemma 3 1B IT: 4 global + 22 SWA attention layers.
PINNED_ATTENTION_LAYERS = 26


def audit_execution_paths(
    file: Path, *, candidate: str, n_ctx: int, root: Path = PROJECT_ROOT,
) -> dict:
    plan = verify_plan(candidate, n_ctx)
    if plan.flash_attention != "on":
        raise ValueError("This precise attention-launch audit requires Flash ON")
    base = inspect_export(file, root, expected_flash="on")
    graph = inspect_graph_replays(
        file, candidate=candidate, n_ctx=n_ctx, root=root)
    out = {
        "status": "cuda_execution_paths_unverified",
        "candidate": candidate,
        "n_ctx": n_ctx,
        "expected_flash_consistent_with_kernel_names":
            base["expected_flash_consistent_with_kernel_names"],
        "cuda_graph_phase_attribution_verified":
            graph["cuda_graph_phase_attribution_verified"],
        "nvtx_phase_ranges": {},
        "per_phase_execution_paths": {},
        "direct_main_kernel_records_correlated": 0,
        "direct_auxiliary_kernel_records_correlated": 0,
        "direct_kernels_unattributed_or_ambiguous": 0,
        "graph_trace_launches_correlated": 0,
        "ranges_with_both_graph_and_direct_attention": 0,
        "ranges_without_dispatch_evidence": 0,
        "ranges_with_unexpected_direct_kernel_counts": 0,
        "execution_path_classification_verified": False,
        "cuda_graph_node_coverage_verified": False,
        "complete_attention_workload_coverage_verified": False,
        "production_qualified": False,
    }
    root = Path(root)
    path = (root / file) if not file.is_absolute() else file
    with sqlite3.connect(path.resolve().as_uri()+"?mode=ro",uri=True) as db:
        names = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"NVTX_EVENTS","StringIds","CUPTI_ACTIVITY_KIND_RUNTIME",
                    "CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_GRAPH_TRACE"}
        if not required <= names:
            return out
        def columns(table):
            return {r[1] for r in db.execute("PRAGMA table_info("+table+")")}
        nc,rc,kc,gc = (
            columns(table) for table in
            ("NVTX_EVENTS","CUPTI_ACTIVITY_KIND_RUNTIME",
             "CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
        )
        if (not {"start","end","eventType","globalTid"} <= nc
            or not {"start","end","globalTid","correlationId","nameId"} <= rc
            or not {"start","end","correlationId"} <= kc
            or "correlationId" not in gc
            or not ("text" in nc or "textId" in nc)):
            return out
        namecol = ("demangledName" if "demangledName" in kc else
                   "shortName" if "shortName" in kc else None)
        if namecol is None:
            return out
        nvtx_label = (
            "COALESCE(n.text,s.value)" if {"text","textId"} <= nc
            else "n.text" if "text" in nc else "s.value")
        nvtx_join = (" LEFT JOIN StringIds s ON n.textId=s.id"
                     if "textId" in nc else "")
        nvtx_sql = (
            "SELECT n.start,n.end,n.globalTid,"+nvtx_label+
            " FROM NVTX_EVENTS n"+nvtx_join+" WHERE n.eventType=59")
        ranges = []
        for start,end,tid,label in db.execute(nvtx_sql):
            if label not in MARKERS.values():
                continue
            if not all(type(x) is int for x in (start,end,tid)) or end <= start:
                return out
            phase = next(p for p,name in MARKERS.items() if label == name)
            ranges.append((start,end,tid,phase))
        ranges.sort()
        phase_count = Counter(x[3] for x in ranges)
        out["nvtx_phase_ranges"] = dict(phase_count)
        expected_ranges = graph["expected_nvtx_phase_ranges"]
        if dict(phase_count) != expected_ranges:
            return out

        # The host CPU launch's entire interval must lie in one same-thread
        # NVTX range. Never compare CPU range timestamps to GPU kernel times.
        def resolve(start,end,tid):
            if (not all(type(x) is int for x in (start,end,tid))
                or end < start):
                return []
            return [i for i,(a,b,t,p) in enumerate(ranges)
                    if t == tid and a <= start and end <= b]

        api = defaultdict(list)
        graph_by_range = Counter()
        graph_unmapped = 0
        for start,end,tid,cid,name in db.execute(
            "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId"
        ):
            api[cid].append((start,end,tid,name))
            if "cudagraphlaunch" not in name.lower():
                continue
            ids = resolve(start,end,tid)
            if len(ids) != 1:
                graph_unmapped += 1
            else:
                graph_by_range[ids[0]] += 1
        # The graph trace must contain exactly one execution row per
        # graph launch (not per CUDA Graph kernel node).
        trace_ids = Counter(row[0] for row in db.execute(
            "SELECT correlationId FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE"))
        graph_launch_cids = {
            cid for cid,events in api.items()
            if any("cudagraphlaunch" in name.lower() for _,_,_,name in events)}
        trace_complete = (len(graph_launch_cids) ==
                          sum(graph_by_range.values())
                          and all(trace_ids[cid] == 1 for cid in graph_launch_cids)
                          and sum(trace_ids.values()) == len(graph_launch_cids))
        out["graph_trace_launches_correlated"] = (
            len(graph_launch_cids) if trace_complete else 0)

        direct_by_range = defaultdict(Counter)
        unattributed = 0
        total_main = 0
        total_helper = 0
        for cid,name in db.execute(
            "SELECT k.correlationId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k."+namecol
        ):
            family = classify_kernel(name)
            if family is None:
                continue
            matched = api.get(cid,[])
            if len(matched) != 1:
                unattributed += 1
                continue
            a,b,t,_ = matched[0]
            ids = resolve(a,b,t)
            if len(ids) != 1:
                unattributed += 1
                continue
            direct_by_range[ids[0]][family] += 1
            if family in MAIN_FLASH:
                total_main += 1
            else:
                total_helper += 1
        out["direct_main_kernel_records_correlated"] = total_main
        out["direct_auxiliary_kernel_records_correlated"] = total_helper
        out["direct_kernels_unattributed_or_ambiguous"] = unattributed

        by_phase = defaultdict(Counter)
        both = missing = unexpected = 0
        for i,(_,_,_,phase) in enumerate(ranges):
            graphs = graph_by_range[i]
            families = direct_by_range[i]
            mains = sum(n for f,n in families.items() if f in MAIN_FLASH)
            helpers = sum(n for f,n in families.items() if f not in MAIN_FLASH)
            if graphs and (mains or helpers):
                by_phase[phase]["mixed_graph_and_direct"] += 1
                both += 1
            elif graphs == 1 and mains == 0 and helpers == 0:
                by_phase[phase]["graph_replay"] += 1
            elif graphs == 0 and mains == PINNED_ATTENTION_LAYERS and (
                helpers == PINNED_ATTENTION_LAYERS):
                by_phase[phase]["direct_attention"] += 1
            elif not graphs and not mains and not helpers:
                by_phase[phase]["no_dispatch_evidence"] += 1
                missing += 1
            else:
                by_phase[phase]["unexpected_dispatch_pattern"] += 1
                unexpected += 1
        out["per_phase_execution_paths"] = {
            phase: dict(by_phase[phase]) for phase in ("prefill","decode")}
        out["ranges_with_both_graph_and_direct_attention"] = both
        out["ranges_without_dispatch_evidence"] = missing
        out["ranges_with_unexpected_direct_kernel_counts"] = unexpected
        verified = (
            graph["cuda_graph_phase_attribution_verified"] is True
            and base["expected_flash_consistent_with_kernel_names"] is True
            and graph_unmapped == 0 and trace_complete and unattributed == 0
            and both == missing == unexpected == 0
            and sum(graph_by_range.values()) ==
                graph["runtime_cuda_graph_launches"]
            and (total_main == total_helper
                 == PINNED_ATTENTION_LAYERS *
                    sum(by_phase[p]["direct_attention"]
                        for p in ("prefill","decode")))
            and all(by_phase[p]["graph_replay"] > 0 and
                    by_phase[p]["direct_attention"] > 0
                    for p in ("prefill","decode"))
        )
        out["execution_path_classification_verified"] = verified
        out["status"] = (
            "cuda_graph_and_direct_launch_paths_verified" if verified else
            "cuda_execution_paths_unverified")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fail-closed offline Phase28 per-range execution-path audit")
    p.add_argument("--sqlite",required=True)
    p.add_argument("--candidate",required=True,choices=("bf16-on","mixed-on"))
    p.add_argument("--n-ctx",required=True,type=int)
    a=p.parse_args(argv)
    try:
        report=audit_execution_paths(
            Path(a.sqlite),candidate=a.candidate,n_ctx=a.n_ctx)
    except (OSError,sqlite3.Error,RuntimeError,ValueError) as exc:
        report={"status":"cuda_execution_paths_unverified",
                "error":type(exc).__name__+": "+str(exc)[:200],
                "execution_path_classification_verified":False,
                "complete_attention_workload_coverage_verified":False,
                "production_qualified":False}
    print(json.dumps(report,sort_keys=True,indent=2))
    return 0 if report["execution_path_classification_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())
