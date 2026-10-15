"""Offline Phase28 Nsight precision-conversion evidence and kernel summary.

Report only explicitly named conversion kernels; names cannot prove all
register-precision operations. Graph phase inference requires same-capture
verified graph kernel -> graph launch -> CPU NVTX join.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_attribution import MARKERS
from .gemma3_phase28_dispatch import (
    classify_kernel, classify_observed_operator, inspect_export)
from .gemma3_phase28_node_audit import inspect_node_trace

CONV_RE = re.compile(
    r"(?<!\w)(convert_unary(?:_cont_vec4)?)\s*"
    r"<\s*([^,<>()]+)\s*,\s*([^,<>()]+)")
TYPES = {
    "float":"F32","float32_t":"F32",
    "__half":"F16","half":"F16",
    "__nv_bfloat16":"BF16","nv_bfloat16":"BF16",
    "__nv_bfloat162":"BF16x2","__half2":"F16x2",
    "double":"F64","int8_t":"I8","int16_t":"I16",
    "int32_t":"I32","int64_t":"I64",
}


def classify_conversion(name: str) -> dict | None:
    m=CONV_RE.search(name)
    if m is None:
        if "convert_unary" not in name:
            return None
        return {"source_dtype":None,"target_dtype":None,
                "recognition":"conversion_kernel_template_unparsed"}
    source,target=(TYPES.get(m.group(k).strip()) for k in (2,3))
    return {
        "source_dtype":source,
        "target_dtype":target,
        "recognition":(
            "source_target_template_verified"
            if source is not None and target is not None
            else "conversion_template_dtype_unrecognized"),
    }


def inspect_precision_trace(
    file: Path, *, candidate: str, n_ctx: int,
    root: Path=PROJECT_ROOT,
) -> dict:
    from .gemma3_phase28_gate import verify_plan
    plan=verify_plan(candidate,n_ctx)
    root=Path(root)
    base=inspect_export(file,root,expected_flash=plan.flash_attention)
    report={
        "status":"precision_trace_unverified",
        "candidate":candidate,"n_ctx":n_ctx,
        "kernel_records_total":0,
        "kernel_records_with_unresolved_name":0,
        "conversion_kernel_records":0,
        "conversion_summary":[],
        "other_precision_name_evidence":[],
        "top_other_kernel_names":[],
        "graph_phase_causal_mapping_verified":False,
        "direct_conversion_phase_correlation_verified":False,
        "graph_node_kernel_coverage_verified":False,
        "source_precision_runtime_inventory_complete":False,
        "production_qualified":False,
        "gpu_inference_executed":False,
    }
    node=None
    path=(root / file) if not file.is_absolute() else file
    path=path.resolve()
    with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
        tabs={r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"CUPTI_ACTIVITY_KIND_KERNEL","StringIds"} <= tabs:
            return report
        kc={r[1] for r in db.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        col=("demangledName" if "demangledName" in kc else
             "shortName" if "shortName" in kc else None)
        if col is None or not {"start","end","correlationId"} <= kc:
            return report
        graph_id_select="k.graphId" if "graphId" in kc else "NULL"
        rows=db.execute(
            "SELECT s.value, k.start,k.end,k.correlationId,"+
            graph_id_select+" FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "LEFT JOIN StringIds s ON s.id=k."+col
        ).fetchall()
        report["kernel_records_total"]=len(rows)
        unresolved=sum(not isinstance(name,str) for name,*_ in rows)
        report["kernel_records_with_unresolved_name"]=unresolved
        if unresolved:
            return report

        # Graph-id phase mapping must be verified in THIS SQLite capture.
        graph_phase={}
        has_node_ids = {"graphNodeId","graphId"} <= kc
        if has_node_ids and any(graphid is not None
                                for _,_,_,_,graphid in rows):
            node=inspect_node_trace(
                file,candidate=candidate,n_ctx=n_ctx,root=root)
            if node.get("graph_to_nvtx_phase_verified") is True:
                graph_phase={int(k):v for k,v in
                             node["graph_id_to_nvtx_phase"].items()}
                report["graph_phase_causal_mapping_verified"]=True
                report["graph_node_kernel_coverage_verified"]=node.get(
                    "graph_kernel_launch_replay_node_sets_verified") is True

        nv_ranges=defaultdict(list)
        runtime={}
        direct_correlations_ambiguous=set()
        if {"NVTX_EVENTS","CUPTI_ACTIVITY_KIND_RUNTIME"} <= tabs:
            nc={r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
            rc={r[1] for r in db.execute(
                "PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")}
            if ({"start","end","globalTid","eventType"} <= nc
                and "text" in nc
                and {"start","end","globalTid","correlationId"} <= rc):
                for start,end,tid,name in db.execute(
                    "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
                    "WHERE eventType=59"):
                    if name not in MARKERS.values():
                        continue
                    phase=next(k for k,v in MARKERS.items() if v==name)
                    nv_ranges[tid].append((start,end,phase))
                for start,end,tid,cid in db.execute(
                    "SELECT start,end,globalTid,correlationId "
                    "FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
                    if cid in runtime:
                        direct_correlations_ambiguous.add(cid)
                    else:
                        matches=set()
                        if all(type(v) is int for v in (start,end,tid)):
                            matches={
                                phase for a,b,phase in nv_ranges.get(tid,())
                                if a <= start <= end <= b}
                        runtime[cid]=(next(iter(matches))
                                      if len(matches)==1 else None)

        conv=defaultdict(lambda:{"records":0,"gpu_ns":0,
                                 "by_phase":Counter(),
                                 "by_graph_id":Counter()})
        other=defaultdict(lambda:{"records":0,"gpu_ns":0})
        fp_name=defaultdict(lambda:{"records":0,"gpu_ns":0})
        unresolved_direct_conv=0
        unknown_phase_graph_conv=0
        for name,start,end,cid,graph_id in rows:
            if (type(start) is not int or type(end) is not int
                or start > end):
                report["status"]="precision_trace_invalid_kernel_time"
                return report
            duration=end-start
            parsed=classify_conversion(name)
            if parsed is not None:
                key=(name[:350],parsed["source_dtype"],
                     parsed["target_dtype"],parsed["recognition"])
                obj=conv[key]
                obj["records"]+=1
                obj["gpu_ns"]+=duration
                if graph_id is None:
                    phase=(runtime.get(cid) if cid not in
                           direct_correlations_ambiguous else None)
                    if phase is None:
                        unresolved_direct_conv+=1
                        phase="unattributed_direct"
                    obj["by_phase"][phase]+=1
                else:
                    obj["by_graph_id"][str(graph_id)]+=1
                    phase=graph_phase.get(graph_id)
                    if phase is None:
                        unknown_phase_graph_conv+=1
                        phase="unattributed_graph"
                    obj["by_phase"][phase]+=1
                continue
            if (classify_kernel(name) is None
                and classify_observed_operator(name) is None):
                obj=other[name[:350]]
                obj["records"]+=1
                obj["gpu_ns"]+=duration
            classification=classify_observed_operator(name)
            if classification is not None:
                obj=fp_name[classification]
                obj["records"]+=1
                obj["gpu_ns"]+=duration
        report["conversion_kernel_records"]=sum(
            x["records"] for x in conv.values())
        report["conversion_summary"]=[
            {"kernel_name":name,"source_dtype":src,
             "target_dtype":dst,"recognition":recognition,
             "records":obj["records"],
             "kernel_gpu_ms_sum":round(obj["gpu_ns"]/1e6,4),
             "by_phase":dict(sorted(obj["by_phase"].items())),
             "by_graph_id":dict(sorted(obj["by_graph_id"].items()))}
            for (name,src,dst,recognition),obj in sorted(
                conv.items(),key=lambda p:-p[1]["gpu_ns"])
        ]
        direct_conversion_count=sum(
            x["by_phase"].get("prefill",0)+x["by_phase"].get("decode",0)+
            x["by_phase"].get("unattributed_direct",0)
            for x in conv.values()) - sum(
                sum(x["by_graph_id"].values()) for x in conv.values())
        # Simpler independent host audit: only claim direct conversion
        # phase attribution when at least one direct conversion was
        # observed and none failed its CUDA runtime + NVTX join.
        report["direct_conversion_phase_correlation_verified"]=bool(
            direct_conversion_count and unresolved_direct_conv==0
            and nv_ranges and runtime and not direct_correlations_ambiguous)
        report["unattributed_direct_conversion_records"]=(
            unresolved_direct_conv)
        report["unattributed_graph_conversion_records"]=(
            unknown_phase_graph_conv)
        report["other_precision_name_evidence"]=[
            {"operator_family":f,"kernel_records":v["records"],
             "kernel_gpu_ms_sum":round(v["gpu_ns"]/1e6,4),
             "limitation":"name evidence only; not full arithmetic dtype"}
            for f,v in sorted(fp_name.items())
        ]
        report["top_other_kernel_names"]=[
            {"kernel_name":k,"kernel_records":v["records"],
             "kernel_gpu_ms_sum":round(v["gpu_ns"]/1e6,4)}
            for k,v in sorted(other.items(),key=lambda p:-p[1]["gpu_ns"])[:25]
        ]
        # Complete as a *name survey*, not as an arithmetic precision audit.
        report["status"]="precision_kernel_name_survey_observed"
        report["kernel_name_survey_verified"]=(
            base["expected_flash_consistent_with_kernel_names"] is True
            and report["kernel_records_total"] == base["kernel_records_total"]
            and unresolved==0)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(
        description="Offline Phase28 CUDA kernel conversion precision survey")
    parser.add_argument("--sqlite",required=True)
    parser.add_argument("--candidate",required=True,
                        choices=("f16-off","bf16-on","f32-off","mixed-on"))
    parser.add_argument("--n-ctx",required=True,type=int)
    args=parser.parse_args(argv)
    try:
        r=inspect_precision_trace(
            Path(args.sqlite),candidate=args.candidate,n_ctx=args.n_ctx)
    except (OSError,ValueError,sqlite3.Error) as exc:
        r={"status":"precision_trace_unverified",
           "error":type(exc).__name__+": "+str(exc)[:200],
           "kernel_name_survey_verified":False,
           "source_precision_runtime_inventory_complete":False,
           "production_qualified":False,"gpu_inference_executed":False}
    print(json.dumps(r,indent=2,sort_keys=True))
    return 0 if r.get("kernel_name_survey_verified") is True else 2


if __name__=="__main__":
    raise SystemExit(main())
