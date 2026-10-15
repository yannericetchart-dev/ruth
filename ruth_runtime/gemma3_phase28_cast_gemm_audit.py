"""Offline Phase28 audit of per-call F32->BF16 casts vs BF16 GEMM launches.

Correlates GPU kernel activity to CPU NVTX intervals via CUDA runtime
correlation IDs, including CUDA Graph node replays in the same SQLite file.
Equal counts and GPU-time ordering do NOT prove producer-consumer buffer
dependencies, GGML operator identity, or attainable speedups.
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
from .gemma3_phase28_node_audit import inspect_node_trace
from .gemma3_phase28_precision_trace import (
    classify_conversion, classify_precision_operator,
)

TARGETS = ("F32_TO_BF16", "BF16_GEMM", "BF16_TO_F16")


def identify_target(name: str) -> str | None:
    cast = classify_conversion(name)
    if cast is not None and cast["recognition"] == "source_target_template_verified":
        if (cast["source_dtype"],cast["target_dtype"]) == ("F32","BF16"):
            return "F32_TO_BF16"
        if (cast["source_dtype"],cast["target_dtype"]) == ("BF16","F16"):
            return "BF16_TO_F16"
    if (classify_precision_operator(name) or (None,))[0] == "GEMM_BF16_NAMED":
        return "BF16_GEMM"
    return None


def audit_cast_gemm(
    file: Path, *, candidate: str, n_ctx: int,
    root: Path = PROJECT_ROOT,
) -> dict:
    plan = verify_plan(candidate, n_ctx)
    if candidate != "bf16-on" or n_ctx != 2048:
        raise ValueError("This audit is pinned to Phase28 bf16-on at context 2048")
    # inspect_export is the common bounded, local Phase28 SQLite gate.
    base = inspect_export(file, root, expected_flash="on")
    nodes = inspect_node_trace(file, candidate=candidate,n_ctx=n_ctx,root=root)
    out = {
        "status":"cast_gemm_range_evidence_unverified",
        "candidate":candidate,"n_ctx":n_ctx,
        "expected_nvtx_ranges":{},
        "nvtx_ranges":{},
        "runtime_graph_launches":0,
        "graph_node_to_nvtx_phase_verified":
            nodes["graph_to_nvtx_phase_verified"],
        "per_phase_path":{},
        "overall_target_kernel_counts":{},
        "unattributed_or_ambiguous_target_kernels":0,
        "target_kernel_records_correlated":0,
        "prefill_per_range_count_balance_verified":False,
        "gpu_timeline_candidate_pairing_verified":False,
        "graph_node_kernel_order_stable":False,
        "graph_node_order_assessed":False,
        "per_range_attribution_verified":False,
        "producer_consumer_buffer_dependency_verified":False,
        "ggml_operator_identity_verified":False,
        "potential_conversion_elimination_verified":False,
        "full_precision_coverage_verified":False,
        "production_qualified":False,
        "gpu_inference_executed":False,
    }
    expected={
        "prefill": (WARMUPS+REPEATS) *
            ((plan.prompt_tokens+plan.n_batch-1)//plan.n_batch),
        "decode": (WARMUPS+REPEATS)*plan.decode_tokens,
    }
    out["expected_nvtx_ranges"]=expected
    if not (base["expected_flash_consistent_with_kernel_names"] is True
            and nodes["graph_to_nvtx_phase_verified"] is True
            and nodes["graph_kernel_launch_replay_node_sets_verified"] is True):
        return out
    root=Path(root)
    fp=((root/file) if not file.is_absolute() else file).resolve()
    with sqlite3.connect(fp.as_uri()+"?mode=ro",uri=True) as db:
        names={x[0] for x in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"NVTX_EVENTS","StringIds",
                "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_KERNEL"} <= names:
            return out
        def cols(table):
            return {r[1] for r in db.execute("PRAGMA table_info("+table+")")}
        nc,rc,kc=(cols(t) for t in (
            "NVTX_EVENTS","CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUPTI_ACTIVITY_KIND_KERNEL"))
        if (not {"start","end","globalTid","eventType"} <= nc
            or not ("text" in nc or "textId" in nc)
            or not {"start","end","globalTid","correlationId","nameId"} <= rc
            or not {"start","end","correlationId",
                    "graphId","graphNodeId"} <= kc):
            return out
        namecol=("demangledName" if "demangledName" in kc else
                 "shortName" if "shortName" in kc else None)
        if not namecol:
            return out
        label=("COALESCE(n.text,s.value)" if {"text","textId"}<=nc
               else "n.text" if "text" in nc else "s.value")
        join=(" LEFT JOIN StringIds s ON n.textId=s.id"
              if "textId" in nc else "")
        nvtx=[]
        for start,end,tid,tag in db.execute(
            "SELECT n.start,n.end,n.globalTid,"+label+
            " FROM NVTX_EVENTS n"+join+" WHERE n.eventType=59"):
            if tag not in MARKERS.values():
                continue
            if not all(type(x) is int for x in (start,end,tid)) or end <= start:
                return out
            phase=next(k for k,v in MARKERS.items() if tag==v)
            nvtx.append((start,end,tid,phase))
        nvtx.sort()
        count=Counter(x[3] for x in nvtx)
        out["nvtx_ranges"]=dict(count)
        if dict(count)!=expected:
            return out

        def locate(start,end,tid):
            if not all(type(x) is int for x in (start,end,tid)) or end<start:
                return []
            return [i for i,(a,b,t,p) in enumerate(nvtx)
                    if t==tid and a<=start<=end<=b]

        apis=defaultdict(list)
        graph_by_range={}
        ambiguous_graph=0
        for start,end,tid,cid,name in db.execute(
            "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN StringIds s ON r.nameId=s.id"):
            apis[cid].append((start,end,tid,name))
            if "cudagraphlaunch" in name.lower():
                indices=locate(start,end,tid)
                if len(indices)!=1 or cid in graph_by_range:
                    ambiguous_graph+=1
                else:
                    graph_by_range[cid]=indices[0]
        out["runtime_graph_launches"]=len(graph_by_range)+ambiguous_graph
        range_graph=defaultdict(list)
        for cid,index in graph_by_range.items():
            range_graph[index].append(cid)
        if (ambiguous_graph or len(graph_by_range)!=
            nodes["runtime_graph_launches_total"] or
            any(len(v)>1 for v in range_graph.values())):
            return out
        if Counter(nvtx[i][3] for i in range_graph)!=Counter(
                nodes["runtime_graph_launches_by_phase"]):
            return out

        events=defaultdict(lambda:defaultdict(list))
        unassigned=0
        alltarget=Counter()
        for name,start,end,cid,gid,node_id in db.execute(
            "SELECT s.value,k.start,k.end,k.correlationId,"
            "k.graphId,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k."+namecol):
            target=identify_target(name)
            if target is None:
                continue
            alltarget[target]+=1
            if (not all(type(t) is int for t in (start,end))
                or end<start):
                unassigned+=1
                continue
            if gid is None and node_id is None:
                ev=apis.get(cid,[])
                # A direct CUDA API must be unique and contained in one
                # CPU NVTX range; do not infer from GPU timestamps.
                index=(locate(ev[0][0],ev[0][1],ev[0][2])
                       if len(ev)==1 and
                       "cudagraphlaunch" not in ev[0][3].lower() else [])
                if len(index)!=1 or index[0] in range_graph:
                    unassigned+=1
                    continue
                range_index=index[0]
                path_type="direct"
            elif gid is not None and node_id is not None:
                range_index=graph_by_range.get(cid)
                if (range_index is None
                    or cid not in apis
                    or not any("cudagraphlaunch" in row[3].lower()
                               for row in apis[cid])
                    or str(gid) not in nodes["graph_id_to_nvtx_phase"]
                    or nodes["graph_id_to_nvtx_phase"][str(gid)] !=
                       nvtx[range_index][3]):
                    unassigned+=1
                    continue
                path_type="graph_node"
            else:
                unassigned+=1
                continue
            events[range_index][target].append(
                (start,end,node_id if path_type=="graph_node" else None))
        out["unattributed_or_ambiguous_target_kernels"]=unassigned
        out["overall_target_kernel_counts"]=dict(alltarget)
        out["target_kernel_records_correlated"]=sum(
            len(v) for group in events.values() for v in group.values())
        stats=defaultdict(lambda:{
            "ranges":0,"target_totals":Counter(),
            "per_range_distributions":defaultdict(Counter),
            "gpu_order_compatible_ranges":0,
            "gpu_order_incompatible_ranges":0,
        })
        balance=True
        order_valid=True
        graph_signatures=defaultdict(set)
        expected_bf16_to_f16=52  # 2 per 26 pinned attention layers
        per_prefill=[]
        for i,(start,end,tid,phase) in enumerate(nvtx):
            mode="graph_node" if i in range_graph else "direct"
            st=stats[(phase,mode)]
            st["ranges"]+=1
            group=events.get(i,{})
            ct={t:len(group.get(t,())) for t in TARGETS}
            for t,num in ct.items():
                st["target_totals"][t]+=num
                st["per_range_distributions"][t][num]+=1
            if phase=="prefill":
                # No equal-count guess is sufficient: check each
                # individual named NVTX range (including all direct).
                ok=ct["F32_TO_BF16"]==ct["BF16_GEMM"] and (
                    ct["F32_TO_BF16"]>0 and
                    ct["BF16_TO_F16"]==expected_bf16_to_f16)
                balance=balance and ok
                per_prefill.append(ct["F32_TO_BF16"])
                conv=sorted(group.get("F32_TO_BF16",()),key=lambda e:(e[0],e[1]))
                gemm=sorted(group.get("BF16_GEMM",()),key=lambda e:(e[0],e[1]))
                compatible=ok and all(
                    c[1]<=g[0] for c,g in zip(conv,gemm))
                st["gpu_order_compatible_ranges"]+=int(compatible)
                st["gpu_order_incompatible_ranges"]+=int(not compatible)
                order_valid=order_valid and compatible
                if mode=="graph_node":
                    signature=(
                        tuple(ev[2] for ev in conv),
                        tuple(ev[2] for ev in gemm))
                    graph_signatures[tuple(range_graph[i])].add(signature)
            else:
                # BF16 GEMM and both casts were observed in prefill
                # only. Any decode instance changes the conclusion.
                if any(ct.values()):
                    balance=False
                    order_valid=False
        report_stats={}
        for (phase,mode),st in sorted(stats.items()):
            report_stats[phase+"_"+mode]={
                "ranges":st["ranges"],
                "target_kernel_totals":dict(st["target_totals"]),
                "per_range_distributions":{
                    t:{str(n):v for n,v in sorted(hist.items())}
                    for t,hist in sorted(st["per_range_distributions"].items())
                },
                "gpu_order_compatible_ranges":
                    st["gpu_order_compatible_ranges"],
                "gpu_order_incompatible_ranges":
                    st["gpu_order_incompatible_ranges"],
            }
        out["per_phase_path"]=report_stats
        out["prefill_f32_bf16_casts_per_range_uniform"]=bool(
            per_prefill and len(set(per_prefill))==1)
        out["prefill_casts_per_range_empirical"]=(
            per_prefill[0] if per_prefill and len(set(per_prefill))==1 else None)
        out["prefill_per_range_count_balance_verified"]=bool(
            balance and out["prefill_f32_bf16_casts_per_range_uniform"]
            and not unassigned)
        # Graph signatures use kernel launch order, not DAG edges.
        # graph signature is required only for prefill graph replay.
        replay_signatures=defaultdict(set)
        for i in range_graph.values():
            if nvtx[i][3]!="prefill":
                continue
            group=events.get(i,{})
            conv=tuple(e[2] for e in sorted(
                group.get("F32_TO_BF16",()),key=lambda v:(v[0],v[1])))
            gemm=tuple(e[2] for e in sorted(
                group.get("BF16_GEMM",()),key=lambda v:(v[0],v[1])))
            # Graph ID comes from correlated kernel metadata; the
            # pinned node audit guarantees one graph per launch.
            graph_ids=set()
            for cid in range_graph[i]:
                graph_ids.update(int(g) for g,p in
                    nodes["graph_id_to_nvtx_phase"].items()
                    if p=="prefill")
            if len(graph_ids)==1:
                replay_signatures[next(iter(graph_ids))].add((conv,gemm))
        out["graph_node_order_assessed"]=bool(replay_signatures)
        out["graph_node_kernel_order_stable"]=bool(
            replay_signatures and
            all(len(sigs)==1 for sigs in replay_signatures.values()))
        out["gpu_timeline_candidate_pairing_verified"]=bool(
            order_valid and not unassigned and
            out["prefill_per_range_count_balance_verified"])
        out["per_range_attribution_verified"]=bool(
            out["prefill_per_range_count_balance_verified"] and
            sum(stat["ranges"] for stat in report_stats.values())==len(nvtx)
            and out["target_kernel_records_correlated"]==sum(alltarget.values())
            and out["runtime_graph_launches"]==
                nodes["runtime_graph_launches_total"])
        if out["per_range_attribution_verified"]:
            out["status"]="cast_and_gemm_per_nvtx_range_accounted"
    return out


def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(
        description="Offline pinned Phase28 cast vs BF16 GEMM per-NVTX audit")
    parser.add_argument("--sqlite",required=True)
    parser.add_argument("--candidate",choices=("bf16-on",),required=True)
    parser.add_argument("--n-ctx",type=int,required=True)
    a=parser.parse_args(argv)
    try:
        r=audit_cast_gemm(
            Path(a.sqlite),candidate=a.candidate,n_ctx=a.n_ctx)
    except (OSError,ValueError,sqlite3.Error) as e:
        r={"status":"cast_gemm_range_evidence_unverified",
           "error":type(e).__name__+": "+str(e)[:200],
           "per_range_attribution_verified":False,
           "producer_consumer_buffer_dependency_verified":False,
           "ggml_operator_identity_verified":False,
           "production_qualified":False}
    print(json.dumps(r,sort_keys=True,indent=2))
    return 0 if r.get("per_range_attribution_verified") else 2


if __name__=="__main__":
    raise SystemExit(main())
