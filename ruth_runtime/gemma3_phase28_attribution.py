"""Fail-closed OFFLINE NVTX -> CUDA API -> CUPTI kernel attribution.

Validated only for the pinned Phase28 workload and native Nsight export.
Never assumes GPU start/end timestamps overlap CPU NVTX push/pop ranges:
kernel launches are associated through CUPTI correlation IDs instead.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_dispatch import (
    classify_kernel, classify_observed_operator, inspect_export,
)
from .gemma3_phase28_gate import WARMUPS, REPEATS, verify_plan

MARKERS = {
    "prefill": "RUTH_GEMMA3_PHASE28_PREFILL",
    "decode": "RUTH_GEMMA3_PHASE28_DECODE",
}
MAIN_FLASH = {"FLASH_VEC", "FLASH_MMA_F16", "FLASH_TILE"}


def inspect_phase_correlations(
    file: Path, *, candidate: str, n_ctx: int,
    root: Path = PROJECT_ROOT,
) -> dict:
    """Correlate kernels using Nsight's NVTX range/CUDA API/CUPTI IDs.

    This is separate from kernel-name inspection: attribution stays false
    unless all expected NVTX ranges and *all* selected attention kernels
    match a unique CPU CUDA-launch record within exactly one named range.
    """
    plan = verify_plan(candidate, n_ctx)
    summary = inspect_export(file, root, expected_flash=plan.flash_attention)
    root = Path(root)
    path = root / file if not file.is_absolute() else file
    result = {
        "status": "cuda_phase_attribution_unverified",
        "candidate": candidate,
        "n_ctx": plan.n_ctx,
        "flash_attention": plan.flash_attention,
        "nvtx_phase_ranges": {},
        "phase_attention_kernel_summary": {},
        "attributed_attention_kernel_records": 0,
        "unattributed_attention_kernel_records": 0,
        "ambiguous_cuda_correlations": 0,
        "prefill_decode_attribution_verified": False,
        "attribution_scope": "recognized_cupti_kernel_launch_records",
        # Correlating every observed kernel does not demonstrate coverage of
        # operations replayed from CUDA Graphs. Require a separate audit.
        "complete_attention_workload_coverage_verified": False,
        "cuda_graph_replay_assessed": False,
        "production_qualified": False,
        "native_kernel_names_verified":
            summary["native_kernel_names_verified"],
        "expected_flash_consistent_with_kernel_names":
            summary["expected_flash_consistent_with_kernel_names"],
    }
    if summary["expected_flash_consistent_with_kernel_names"] is not True:
        return result
    path = path.resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        names = {x[0] for x in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"} <= names:
            return result
        def cols(table):
            return {r[1] for r in db.execute("PRAGMA table_info("+table+")")}
        nvcols = cols("NVTX_EVENTS")
        rtcols = cols("CUPTI_ACTIVITY_KIND_RUNTIME")
        kcols = cols("CUPTI_ACTIVITY_KIND_KERNEL")
        if (not {"start", "end", "eventType", "globalTid"} <= nvcols
            or not ({"start", "end", "globalTid", "correlationId"} <= rtcols)
            or not {"start", "end", "correlationId"} <= kcols):
            return result
        namecol = "demangledName" if "demangledName" in kcols else (
            "shortName" if "shortName" in kcols else None)
        if not namecol or not ("text" in nvcols or "textId" in nvcols):
            return result
        label = ("COALESCE(n.text, s.value)" if
                 {"text", "textId"} <= nvcols else
                 "n.text" if "text" in nvcols else "s.value")
        join_text = (" LEFT JOIN StringIds s ON n.textId=s.id" if
                     "textId" in nvcols else "")
        nv_sql = (
            "SELECT n.start,n.end,n.globalTid,"+label+
            " FROM NVTX_EVENTS n"+join_text+" WHERE n.eventType=59"
        )
        by_tid = defaultdict(list)
        range_counts = Counter()
        for start, end, tid, text in db.execute(nv_sql):
            if text not in MARKERS.values():
                continue
            if (type(start) is not int or type(end) is not int
                or type(tid) is not int or end <= start):
                return result
            phase = next(k for k,v in MARKERS.items() if v == text)
            by_tid[tid].append((start,end,phase))
            range_counts[phase] += 1
        result["nvtx_phase_ranges"] = dict(range_counts)
        expected_ranges = {
            "prefill": (WARMUPS+REPEATS) *
                ((plan.prompt_tokens + plan.n_batch - 1)//plan.n_batch),
            "decode": (WARMUPS+REPEATS) * plan.decode_tokens,
        }
        if dict(range_counts) != expected_ranges:
            result["expected_nvtx_phase_ranges"] = expected_ranges
            return result
        # Use the documented Nsight Systems kernel correlationId -> CUDA
        # runtime API correlationId join. globalTid and whole runtime API
        # interval must fit the NVTX CPU range. No GPU-timestamp heuristic.
        query = (
            "SELECT k.rowid, k.start, k.end, s.value, "
            "r.start, r.end, r.globalTid "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k."+namecol+"=s.id "
            "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r "
            "ON r.correlationId=k.correlationId "
            "WHERE k.end>=k.start"
        )
        matched = defaultdict(set)
        selected = {}
        for kid,kstart,kend,kname,rstart,rend,tid in db.execute(query):
            family = classify_kernel(kname)
            if family is None and plan.flash_attention == "off":
                if classify_observed_operator(kname) == "SOFTMAX_F32":
                    family = "SOFTMAX_F32"
            if family is None:
                continue
            selected[kid] = (family,kend-kstart)
            if all(type(x) is int for x in (rstart,rend,tid)) and rstart <= rend:
                matches = {phase for begin,end,phase in by_tid.get(tid,())
                           if begin <= rstart and rend <= end}
                matched[kid].update(matches)
                # Repeated runtime correlation means the kernel-to-API
                # association is ambiguous even if phase names agree.
                if matches:
                    matched[kid].add("MATCHED_API")
        # We deliberately do not count a kernel as verified if its
        # correlation resolved to more than one runtime API launch.
        # (The API-match audit below separately checks duplicates.)
        api_duplicates = set()
        seen_apis = Counter()
        for kid, kstart, kend, kname, rstart, rend, tid in db.execute(query):
            if kid in selected and rstart is not None:
                seen_apis[kid] += 1
                if seen_apis[kid] > 1:
                    api_duplicates.add(kid)
        per_phase = defaultdict(lambda: defaultdict(lambda: [0,0]))
        unresolved = 0
        ambiguous = 0
        for kid,(family,ns) in selected.items():
            phases = matched[kid] - {"MATCHED_API"}
            if kid in api_duplicates or len(phases) > 1:
                ambiguous += 1
                continue
            if len(phases) != 1:
                unresolved += 1
                continue
            phase = next(iter(phases))
            per_phase[phase][family][0] += 1
            per_phase[phase][family][1] += ns
        result["ambiguous_cuda_correlations"] = ambiguous
        result["unattributed_attention_kernel_records"] = unresolved + ambiguous
        result["attributed_attention_kernel_records"] = (
            len(selected) - unresolved - ambiguous)
        result["phase_attention_kernel_summary"] = {
            phase: [
                {"family": f, "count": c, "kernel_gpu_ms": round(ns/1e6,4)}
                for f,(c,ns) in sorted(families.items())
            ]
            for phase,families in sorted(per_phase.items())
        }
        principal = (MAIN_FLASH if plan.flash_attention == "on"
                     else {"SOFTMAX_F32"})
        phases_with_principal = {
            phase for phase,families in per_phase.items()
            if any(f in families for f in principal)
        }
        verified = bool(selected) and not unresolved and not ambiguous and (
            phases_with_principal == set(MARKERS))
        result["prefill_decode_attribution_verified"] = verified
        result["status"] = (
            "cuda_phase_attribution_verified" if verified else
            "cuda_phase_attribution_unverified"
        )
        return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Offline Nsight NVTX/CUPTI Phase28 kernel correlation")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--candidate",
                   choices=("f16-off","bf16-on","f32-off","mixed-on"),
                   required=True)
    p.add_argument("--n-ctx", type=int, required=True)
    args = p.parse_args(argv)
    try:
        report = inspect_phase_correlations(
            Path(args.sqlite), candidate=args.candidate, n_ctx=args.n_ctx)
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        report = {"status":"cuda_phase_attribution_unverified",
                  "error":type(exc).__name__+": "+str(exc)[:200],
                  "prefill_decode_attribution_verified":False,
                  "production_qualified":False}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["prefill_decode_attribution_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
