"""Offline ranking of tensor subsets from existing Ruth Phase-2 measurements.

No CUDA calls. Measurements at the *same* profiling context determine
tensor-level memory/speed effects, and empirical pair/triple interactions
correct non-additive GPU allocation effects. Rank candidates for real GPU
verification at the target context; never assume predicted fit is proof.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import statistics
from tempfile import NamedTemporaryFile


SCHEMA = "ruth-phase2-offload-candidate-plan-v1"


def plan_from_report(report: dict, *, source_sha256: str, target_ctx: int = 512,
                     min_tensors: int = 2, max_tensors: int = 4,
                     shortlist: int = 100, decode_weight: float = .7) -> dict:
    if report.get("schema") != "ruth-phase2-adaptive-tensor-greedy-v3" or not report.get("completed"):
        raise ValueError("Serve un report greedy v3 confermato.")
    if (not 1 <= min_tensors <= max_tensors <= 5
        or shortlist < 1 or not 0 <= decode_weight <= 1):
        raise ValueError("Parametri piano invalidi.")
    stage = next((stage for stage in report.get("stages", [])
                  if stage.get("successful") and stage.get("measured_target_n_ctx") == target_ctx), None)
    if stage is None:
        raise ValueError("Target senza seed confermata.")
    seed = tuple(sorted(stage["confirmed_host_tensor_names"]))
    confirmed = stage["validation"]
    if (confirmed.get("success") is not True or confirmed.get("repetitions", 0) < 3
        or tuple(sorted(confirmed.get("host_tensor_names", []))) != seed
        or not all(s.get("native_placement_verified") is True
                   and s.get("gpu_kv_verified") is True
                   for s in confirmed.get("samples", []))):
        raise ValueError("Seed non confermata nativamente.")
    anchor = stage["from_n_ctx"]
    measured = {
        tuple(sorted(row["host_tensor_names"])): row
        for row in report.get("measured_configs", {}).values()
        if (row.get("n_ctx") == anchor and row.get("success") is True
            and isinstance(row.get("metrics"), dict))
    }
    base = measured.get(())
    seed_anchor = measured.get(seed)
    if not base or not seed_anchor:
        raise ValueError("Mancano baseline o seed misurate all'anchor.")
    singles = {names[0]: r for names, r in measured.items() if len(names) == 1}
    groups = defaultdict(list)
    for name in singles:
        groups[name.split(".", 2)[-1]].append(name)
    if not groups:
        raise ValueError("Nessun tensore individuale misurato.")
    family, names = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
    names = sorted(names)
    if len(names) < min_tensors:
        raise ValueError("Numero di tensori individualmente misurati insufficiente.")
    control = base["metrics"]
    # Independent effects of moving exactly ONE GGUF tensor.
    effect = {}
    for name in names:
        row = singles[name]["metrics"]
        effect[name] = {
            "saved": control["vram_peak_mib"] - row["vram_peak_mib"],
            "prefill": math.log(control["prefill_tok_s"] / row["prefill_tok_s"]),
            "decode": math.log(control["decode_tok_s"] / row["decode_tok_s"]),
        }
    # Median deviations from additivity, measured on real pairs/triples.
    interactions = defaultdict(lambda: defaultdict(list))
    for subset, row in measured.items():
        if len(subset) not in (2, 3) or not set(subset).issubset(effect):
            continue
        observed = row["metrics"]
        interactions[len(subset)]["saved"].append(
            control["vram_peak_mib"] - observed["vram_peak_mib"]
            - sum(effect[t]["saved"] for t in subset))
        for label in ("prefill", "decode"):
            interactions[len(subset)][label].append(
                math.log(control[label + "_tok_s"] / observed[label + "_tok_s"])
                - sum(effect[t][label] for t in subset))
    med = lambda n, metric: statistics.median(interactions[n][metric]) if interactions[n][metric] else 0.
    bias = {}
    for n in range(min_tensors, max_tensors + 1):
        bias[n] = {
            key: (
                med(n, key) if interactions[n][key] else
                (n - 2) * med(3, key) - (n - 3) * med(2, key) if n > 3 else 0
            ) for key in ("saved", "prefill", "decode")
        }
    seed_512 = confirmed["metrics"]
    seed_256 = seed_anchor["metrics"]
    allocation_step = seed_512["vram_peak_mib"] - seed_256["vram_peak_mib"]
    factors = {key: seed_512[key] / seed_256[key]
               for key in ("prefill_tok_s", "decode_tok_s")}
    total = seed_512["vram_total_mib"]

    ranked = []
    considered = 0
    for count in range(min_tensors, max_tensors + 1):
        for subset in combinations(names, count):
            considered += 1
            direct = measured.get(subset)
            if direct:
                observed = direct["metrics"]
                saved = control["vram_peak_mib"] - observed["vram_peak_mib"]
                speeds = {key: observed[key] for key in factors}
            else:
                saved = sum(effect[n]["saved"] for n in subset) + bias[count]["saved"]
                speeds = {
                    label + "_tok_s": control[label + "_tok_s"] * math.exp(
                        -sum(effect[name][label] for name in subset) - bias[count][label])
                    for label in ("prefill", "decode")
                }
            prefill = speeds["prefill_tok_s"] * factors["prefill_tok_s"]
            decode = speeds["decode_tok_s"] * factors["decode_tok_s"]
            peak = control["vram_peak_mib"] - saved + allocation_step
            headroom = total - peak
            score = decode_weight * math.log(decode) + (1 - decode_weight) * math.log(prefill)
            ranked.append({
                "host_tensor_names": list(subset), "n_ctx": target_ctx,
                "observed_at_anchor": bool(direct),
                "estimation_basis": ("direct_anchor_observation_plus_seed_ctx_transfer"
                                     if direct else "singleton_plus_empirical_interaction_medians"),
                "estimated_vram_freed_at_anchor_mib": round(saved, 3),
                "estimated_peak_vram_at_target_mib": round(peak, 3),
                "estimated_headroom_at_target_mib": round(headroom, 3),
                "estimated_prefill_tok_s_at_target": round(prefill, 3),
                "estimated_decode_tok_s_at_target": round(decode, 3),
                "estimated_speed_objective": round(score, 8),
                "estimated_fits_physical_vram": headroom > 0,
            })
    # Inferred physical fit is a SORT preference, not a minimum-headroom gate.
    ranked.sort(key=lambda r: (
        not r["estimated_fits_physical_vram"], -r["estimated_speed_objective"],
        len(r["host_tensor_names"]), r["host_tensor_names"]
    ))
    selected = ranked[:shortlist]
    for i, row in enumerate(selected, 1):
        row["rank"] = i
    return {
        "schema": SCHEMA, "source_greedy_sha256": source_sha256,
        "model_id": report["parameters"]["model_id"],
        "model_path": report["parameters"]["model_path"],
        "target_n_ctx": target_ctx, "anchor_n_ctx": anchor,
        "seed_host_tensor_names": list(seed),
        "seed_confirmed_target_metrics": seed_512,
        "planning_parameters": {
            "min_tensors": min_tensors, "max_tensors": max_tensors,
            "shortlist": shortlist, "decode_weight": decode_weight,
            "candidate_family": family,
            "single_tensor_observations": len(names),
            "pair_observations": len(interactions[2]["saved"]),
            "triple_observations": len(interactions[3]["saved"]),
            "total_combinations_considered": considered,
        },
        "calibration": {
            "baseline_256_peak_mib": control["vram_peak_mib"],
            "observed_ctx_peak_delta_256_to_512_mib": round(allocation_step, 3),
            "seed_speed_ratio_256_to_512": factors,
            "empirical_interaction_medians_by_group_size": bias,
        },
        "candidates": selected,
        "disclaimer": (
            "Speed and VRAM are approximate estimates; measured GPU savings "
            "are non-additive. No minimum VRAM margin is applied. "
            "Each candidate needs independent native GPU-KV CUDA validation."
        ),
    }


def generate_plan(source_report: Path, output: Path, **kwargs) -> dict:
    source_report = Path(source_report).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if source_report == output:
        raise ValueError("Il piano non deve sovrascrivere il report sorgente.")
    raw = source_report.read_bytes()
    report = json.loads(raw)
    plan = plan_from_report(report, source_sha256=hashlib.sha256(raw).hexdigest(),
                            **kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=output.parent,
                            prefix=output.name + ".", suffix=".tmp",
                            encoding="utf-8", delete=False) as temp:
        json.dump(plan, temp, ensure_ascii=False, indent=2, allow_nan=False)
        temp.write("\n")
        path = Path(temp.name)
    path.replace(output)
    return plan


def main():
    parser = argparse.ArgumentParser(description="Genera offline la lista delle combinazioni Ruth.")
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("phase2_candidate_plan_512.json"))
    parser.add_argument("--target-ctx", type=int, default=512)
    parser.add_argument("--min-tensors", type=int, default=2)
    parser.add_argument("--max-tensors", type=int, default=4)
    parser.add_argument("--shortlist", type=int, default=100)
    parser.add_argument("--decode-weight", type=float, default=.7)
    args = parser.parse_args()
    try:
        report = generate_plan(
            args.source_report, args.output, target_ctx=args.target_ctx,
            min_tensors=args.min_tensors, max_tensors=args.max_tensors,
            shortlist=args.shortlist, decode_weight=args.decode_weight
        )
        print(json.dumps({
            "output": str(args.output),
            "combinations_considered": report["planning_parameters"]["total_combinations_considered"],
            "candidates_selected": len(report["candidates"]),
            "top_candidates": report["candidates"][:3],
        }, indent=2))
    except Exception as e:
        print(f"Planner: {type(e).__name__}: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
