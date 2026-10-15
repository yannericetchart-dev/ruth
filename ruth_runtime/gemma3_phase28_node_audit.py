"""Offline audit of observed CUDA Graph kernel-node replay multiplicities.

This reader covers CUDA kernel records only; it does not assert that all
CUDA Graph node types (memcpy, memset, events, host) have been captured.
It never attributes a graph to an NVTX phase by count alone.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_attribution import MARKERS
from .gemma3_phase28_dispatch import classify_kernel, inspect_export
from .gemma3_phase28_gate import REPEATS, WARMUPS, verify_plan


def inspect_node_trace(
    file: Path, *, candidate: str, n_ctx: int, root: Path = PROJECT_ROOT,
) -> dict:
    plan = verify_plan(candidate, n_ctx)
    base = inspect_export(file, root, expected_flash=plan.flash_attention)
    report = {
        "status": "graph_node_repetition_unverified",
        "candidate": candidate,
        "n_ctx": n_ctx,
        "expected_flash_consistent_with_kernel_names":
            base["expected_flash_consistent_with_kernel_names"],
        "kernel_records_total": 0,
        "kernel_records_with_graph_and_node_id": 0,
        "kernel_records_without_graph_and_node_id": 0,
        "kernel_records_with_partial_graph_identity": 0,
        "nvtx_phase_ranges": {},
        "runtime_graph_launches_by_phase": {},
        "runtime_graph_launches_total": 0,
        "unmatched_or_ambiguous_graph_launches": 0,
        "graph_node_kernel_summary": [],
        "graph_phase_candidates_by_count": [],
        "graph_node_kernel_families": [],
        "observed_graph_kernel_node_repetitions_verified": False,
        "graph_to_nvtx_phase_verified": False,
        "all_cuda_graph_node_types_traced_verified": False,
        "complete_attention_workload_coverage_verified": False,
        "production_qualified": False,
    }
    path = (root / file) if not file.is_absolute() else file
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro",uri=True) as db:
        names = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds",
                    "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME"}
        if not required <= names:
            return report
        def cols(name):
            return {x[1] for x in db.execute("PRAGMA table_info("+name+")")}
        kc,nc,rc = (
            cols(name) for name in ("CUPTI_ACTIVITY_KIND_KERNEL",
                                    "NVTX_EVENTS","CUPTI_ACTIVITY_KIND_RUNTIME"))
        if (not {"graphId","graphNodeId","start","end"} <= kc
            or not {"start","end","eventType","globalTid"} <= nc
            or not {"start","end","globalTid","nameId"} <= rc
            or not ("text" in nc or "textId" in nc)):
            return report
        namecol = ("demangledName" if "demangledName" in kc else
                   "shortName" if "shortName" in kc else None)
        if not namecol:
            return report

        counts = db.execute("""
            SELECT COUNT(*),
                SUM(CASE WHEN graphId IS NOT NULL
                          AND graphNodeId IS NOT NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN graphId IS NULL
                          AND graphNodeId IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN (graphId IS NULL) != (graphNodeId IS NULL)
                          THEN 1 ELSE 0 END)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        """).fetchone()
        report["kernel_records_total"] = counts[0]
        report["kernel_records_with_graph_and_node_id"] = counts[1] or 0
        report["kernel_records_without_graph_and_node_id"] = counts[2] or 0
        report["kernel_records_with_partial_graph_identity"] = counts[3] or 0

        # Every CUDA runtime graph-launch API is attributed to exactly one
        # same-thread CPU NVTX range using both interval endpoints.
        label = (
            "COALESCE(n.text,s.value)" if {"text","textId"} <= nc
            else "n.text" if "text" in nc else "s.value")
        nvjoin = (" LEFT JOIN StringIds s ON n.textId=s.id"
                  if "textId" in nc else "")
        ranges = defaultdict(list)
        nvtx_counts = Counter()
        sql = ("SELECT n.start,n.end,n.globalTid,"+label+
               " FROM NVTX_EVENTS n"+nvjoin+" WHERE n.eventType=59")
        for start,end,tid,tag in db.execute(sql):
            if tag not in MARKERS.values():
                continue
            if not all(type(x) is int for x in (start,end,tid)) or end <= start:
                return report
            phase = next(p for p,mark in MARKERS.items() if tag == mark)
            ranges[tid].append((start,end,phase))
            nvtx_counts[phase] += 1
        report["nvtx_phase_ranges"] = dict(nvtx_counts)
        phase_launches = Counter()
        bad_launches = 0
        for start,end,tid,name in db.execute(
            "SELECT r.start,r.end,r.globalTid,s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON s.id=r.nameId "
            "WHERE LOWER(s.value) LIKE '%cudagraphlaunch%'"
        ):
            matched = set()
            if all(type(x) is int for x in (start,end,tid)) and start <= end:
                matched = {phase for lo,hi,phase in ranges.get(tid,())
                           if lo <= start and end <= hi}
            if len(matched) != 1:
                bad_launches += 1
            else:
                phase_launches[next(iter(matched))] += 1
        report["runtime_graph_launches_by_phase"] = dict(phase_launches)
        report["runtime_graph_launches_total"] = (
            sum(phase_launches.values())+bad_launches)
        report["unmatched_or_ambiguous_graph_launches"] = bad_launches

        # Per-(graph, node) frequencies are the key observed replay
        # evidence. This does not require GRAPH_TRACE in node mode.
        node_count = db.execute("""
            SELECT graphId,graphNodeId,COUNT(*)
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE graphId IS NOT NULL AND graphNodeId IS NOT NULL
            GROUP BY graphId,graphNodeId
        """).fetchall()
        nodes_by_graph = defaultdict(list)
        for graph_id,node_id,count in node_count:
            nodes_by_graph[graph_id].append(count)

        main_rows = []
        all_uniform = True
        for graph_id, values in sorted(nodes_by_graph.items()):
            histogram = Counter(values)
            uniform = len(histogram) == 1 and values[0] > 0
            all_uniform = all_uniform and uniform
            main_rows.append({
                "graph_id": graph_id,
                "unique_kernel_node_ids": len(values),
                "kernel_records": sum(values),
                "node_launch_frequency_min": min(values),
                "node_launch_frequency_max": max(values),
                "node_launch_frequency_histogram": {
                    str(n): count for n,count in sorted(histogram.items())},
                "every_observed_kernel_node_repeated_equally": uniform,
            })
        report["graph_node_kernel_summary"] = main_rows

        # Counts give possible phase labels but not a causal
        # graphId-to-NVTX correlation. Preserve that distinction.
        candidate_matches = []
        unique_phase_counts = Counter(phase_launches.values())
        unique_graph_counts = Counter(
            row["node_launch_frequency_min"] for row in main_rows
            if row["every_observed_kernel_node_repeated_equally"])
        for entry in main_rows:
            repetitions = entry["node_launch_frequency_min"]
            possible_phases = sorted(
                phase for phase,n in phase_launches.items()
                if n == repetitions)
            candidate_matches.append({
                "graph_id": entry["graph_id"],
                "candidate_nvtx_phases_by_count_only": possible_phases,
                "repeat_count": repetitions,
            })
        report["graph_phase_candidates_by_count"] = candidate_matches

        families = defaultdict(Counter)
        for graph_id,name,count in db.execute(
            "SELECT k.graphId,s.value,COUNT(*) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k."+namecol+"=s.id "
            "WHERE k.graphId IS NOT NULL AND k.graphNodeId IS NOT NULL "
            "GROUP BY k.graphId,s.value"
        ):
            fam = classify_kernel(name) or "NON_FLASH_OTHER"
            families[graph_id][fam] += count
        report["graph_node_kernel_families"] = [
            {"graph_id":gid,
             "kernel_families":dict(sorted(families[gid].items()))}
            for gid in sorted(nodes_by_graph)
        ]

        expected_ranges = {
            "prefill": (WARMUPS+REPEATS)*
                ((plan.prompt_tokens+plan.n_batch-1)//plan.n_batch),
            "decode": (WARMUPS+REPEATS)*plan.decode_tokens,
        }
        count_match = (
            len(main_rows) == len(phase_launches) == 2
            and all(unique_phase_counts[n] == 1
                    for n in phase_launches.values())
            and unique_graph_counts == unique_phase_counts)
        verified = (
            base["expected_flash_consistent_with_kernel_names"] is True
            and dict(nvtx_counts) == expected_ranges
            and bad_launches == 0
            and report["kernel_records_with_graph_and_node_id"] > 0
            and report["kernel_records_with_partial_graph_identity"] == 0
            and all_uniform and count_match
        )
        report["observed_graph_kernel_node_repetitions_verified"] = verified
        report["status"] = (
            "observed_graph_kernel_node_repetition_counts_verified"
            if verified else "graph_node_repetition_unverified"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Read-only Phase28 CUDA Graph kernel-node repetitions")
    p.add_argument("--sqlite",required=True)
    p.add_argument("--candidate",required=True,choices=(
        "f16-off","bf16-on","f32-off","mixed-on"))
    p.add_argument("--n-ctx",required=True,type=int)
    args = p.parse_args(argv)
    try:
        report = inspect_node_trace(
            Path(args.sqlite),candidate=args.candidate,n_ctx=args.n_ctx)
    except (OSError,sqlite3.Error,RuntimeError,ValueError) as exc:
        report = {
            "status":"graph_node_repetition_unverified",
            "error":type(exc).__name__+": "+str(exc)[:200],
            "observed_graph_kernel_node_repetitions_verified":False,
            "graph_to_nvtx_phase_verified":False,
            "complete_attention_workload_coverage_verified":False,
            "production_qualified":False,
        }
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if report["observed_graph_kernel_node_repetitions_verified"] else 2


if __name__=="__main__":
    raise SystemExit(main())
