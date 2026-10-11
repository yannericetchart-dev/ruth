"""Bounded local screen around an empirically confirmed granular mapping.

Every proposed extension comes from the GGUF inventory, across all families
that have a host->GPU single-tensor alternative. Diagnostics only: the seed
is experimental, n_ctx=512 is not operational, and no profile is selected.
"""

import argparse
from collections import defaultdict
import json
import math
import re
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark
from .mapping_candidates import propose_mapping_candidates
from .mapping_composition import compose_candidates
from .measured_combinations import choose_compound_trials


def choose_neighbors(proposals: dict, seed_ids: list[str],
                     candidate_count: int = 3) -> list[dict]:
    """Choose non-overlapping, per-family-diverse GPU promotions.

    Prefer tensors in blocks not already changed by the seed, then smaller
    GGUF weights to avoid unnecessarily large initial allocations. This
    selection orders *experiments*, not mappings by expected performance.
    No list of allowed families or performance constants is hard-coded.
    """
    if not isinstance(candidate_count, int) or candidate_count < 1:
        raise ValueError("candidate_count deve essere positivo.")
    seed = compose_candidates(proposals, seed_ids)
    used_names = set(seed["tensor_names"])
    seed_blocks = {
        int(match.group(1))
        for name in used_names
        if (match := re.match(r"blk\.(\d+)\.", name))
    }
    groups = defaultdict(list)
    for candidate in proposals["candidates"]:
        if (candidate["direction"] != "host_to_gpu"
            or candidate["kind"] != "single_tensor"
            or len(candidate["tensor_names"]) != 1
            or candidate["tensor_names"][0] in used_names
            or not isinstance(candidate.get("gguf_bytes_to_gpu"), int)
            or candidate["gguf_bytes_to_gpu"] <= 0):
            continue
        groups[candidate["family"]].append(candidate)
    for family, choices in groups.items():
        choices.sort(key=lambda c: (
            1 if any(
                re.match(r"blk\." + str(b) + r"\.", c["tensor_names"][0])
                for b in seed_blocks
            ) else 0,
            c["gguf_bytes_to_gpu"],
            c["id"],
        ))

    selected = []
    # One candidate per available family before selecting a second in any.
    while len(selected) < candidate_count and any(groups.values()):
        for family in sorted(groups):
            if groups[family]:
                selected.append(groups[family].pop(0))
                if len(selected) >= candidate_count:
                    break
    return selected


def summarize_neighborhood(trials: list[dict], neighbors: list[dict],
                           runs: int) -> list[dict]:
    ids = ["seed"] + [c["id"] for c in neighbors]
    if len(set(ids)) != len(ids):
        raise ValueError("ID esperimento duplicati.")
    seen = {}
    for trial in trials:
        key = (trial["run"], trial["id"])
        if (not isinstance(trial.get("run"), int)
            or trial["run"] not in range(1, runs + 1)
            or trial.get("id") not in ids or key in seen):
            raise ValueError("Ripetizione sconosciuta o duplicata.")
        if trial.get("success") is not True:
            raise ValueError("Prova non valida nella sintesi.")
        if trial["id"] != "seed" and trial.get("mapping_verified") is not True:
            raise ValueError("Mapping non confermato nativamente.")
        for metric in ("decode_tps", "prefill_tps", "vram_peak_mib"):
            value = trial.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Misura assente o non finita: {metric}")
        if not isinstance(trial.get("generated_tokens"), int) or trial["generated_tokens"] < 48:
            raise ValueError("Campione decode troppo corto.")
        seen[key] = trial
    if len(seen) != len(ids) * runs:
        raise ValueError("Ripetizioni mancanti.")
    results = []
    for ident in ids:
        rows = [seen[(run, ident)] for run in range(1, runs + 1)]
        pairs = [
            {
                "run": run,
                "decode_delta_tps": round(
                    seen[(run, ident)]["decode_tps"] - seen[(run, "seed")]["decode_tps"], 3
                ),
                "prefill_delta_tps": round(
                    seen[(run, ident)]["prefill_tps"] - seen[(run, "seed")]["prefill_tps"], 3
                ),
                "vram_peak_delta_mib": round(
                    seen[(run, ident)]["vram_peak_mib"] - seen[(run, "seed")]["vram_peak_mib"], 1
                ),
            }
            for run in range(1, runs + 1)
        ]
        results.append({
            "id": ident,
            "family": next((c["family"] for c in neighbors if c["id"] == ident), "seed"),
            "median_decode_tps": round(statistics.median(t["decode_tps"] for t in rows), 2),
            "median_prefill_tps": round(statistics.median(t["prefill_tps"] for t in rows), 2),
            "max_sampled_vram_peak_mib": max(t["vram_peak_mib"] for t in rows),
            "paired_differences": pairs,
            "decode_faster_pairs": sum(p["decode_delta_tps"] > 0 for p in pairs),
            "prefill_faster_pairs": sum(p["prefill_delta_tps"] > 0 for p in pairs),
        })
    return results


def screen(gpu_layers: int = 40, candidate_count: int = 3,
           runs: int = 3, tokens: int = 128, timeout: int = 180) -> dict:
    if (not 1 <= candidate_count <= 12 or not 2 <= runs <= 5
        or not 64 <= tokens <= 256 or timeout < 1):
        raise ValueError("candidate_count=1..12, runs=2..5, tokens=64..256, timeout>=1.")

    initial = GPUManager.calibration_status()
    if not initial["ready"]:
        raise RuntimeError(f"GPU non pronta: {initial['errors']}; {initial['foreign_processes']}")

    proposals = propose_mapping_candidates(
        ModelManager().inspect("gemma-2-9b-it", n_ctx=512), gpu_layers
    )
    seed = next(
        c for c in choose_compound_trials(proposals) if c["id"] == "dual_promotion"
    )
    compose_candidates(proposals, seed["component_ids"])
    neighbors = choose_neighbors(proposals, seed["component_ids"], candidate_count)
    if not neighbors:
        raise ValueError("Nessuna estensione host->GPU disgiunta disponibile.")

    variants = [{"id": "seed", "component_ids": seed["component_ids"]}]
    variants += [{
        "id": c["id"], "component_ids": seed["component_ids"] + [c["id"]],
        "added_tensor": c["tensor_names"][0],
        "family": c["family"],
        "estimated_extra_gguf_bytes": c["gguf_bytes_to_gpu"],
    } for c in neighbors]

    report = {
        "phase": "confirmed_seed_granular_neighborhood",
        "model": "gemma-2-9b-it",
        "backend": "cuda",
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "seed_source": "previously_native_verified_dual_promotion",
        "seed_component_ids": seed["component_ids"],
        "candidate_inventory_count": proposals["candidate_count"],
        "experiment_selection": "family_round_robin_then_smaller_weights_outside_seed_blocks",
        "runs_per_variant": runs,
        "tokens": tokens,
        "gpu_total_mib": initial["memory"]["total_mib"],
        "variants": variants,
        "trials": [],
        "summary": [],
        "success": False,
        "profile_generated": False,
        "note": (
            "Ricerca locale: estensioni scelte da inventario GGUF senza "
            "presumere vantaggi prestazionali. Ogni confronto e' appaiato "
            "al seed misurato nello stesso giro; log nativi obbligatori. "
            "Non sono prove di qualita' o di significativita' statistica; "
            "nessun profilo o contesto operativo selezionato."
        ),
    }
    for iteration in range(1, runs + 1):
        order = variants if iteration % 2 else variants[::-1]
        for variant in order:
            state = GPUManager.calibration_status()
            if not state["ready"]:
                report["error"] = (
                    f"GPU modificata durante la ricerca: {state['errors']}; "
                    f"{state['foreign_processes']}"
                )
                return report
            observed = benchmark(
                gpu_layers=gpu_layers, max_tokens=tokens, timeout=timeout,
                combination_ids=variant["component_ids"],
            )
            metrics = observed.get("measurements") or {}
            verification = observed.get("mapping_verification") or {}
            valid = observed.get("success") is True and verification.get("verified") is True
            row = {
                "run": iteration,
                "id": variant["id"],
                "success": valid,
                "mapping_verified": verification.get("verified"),
                "decode_tps": metrics.get("decode_tokens_per_second"),
                "prefill_tps": metrics.get("prefill_tokens_per_second"),
                "vram_peak_mib": metrics.get("vram_peak_sampled_mib"),
                "generated_tokens": metrics.get("generated_tokens"),
                "worker_returncode": observed.get("worker_returncode"),
                "failure_type": observed.get("failure_type"),
                "native_signal": observed.get("native_signal"),
                "error": observed.get("error"),
            }
            report["trials"].append(row)
            if not valid:
                report["error"] = (
                    f"Stop al candidato {variant['id']}, giro {iteration}: "
                    f"{row['error'] or row['failure_type'] or 'log nativi non confermati'}"
                )
                return report

    report["summary"] = summarize_neighborhood(report["trials"], neighbors, runs)
    report["success"] = True
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Esplorazione locale di tutte le famiglie GGUF sul seed confermato"
    )
    parser.add_argument("--gpu-layers", type=int, default=40)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        result = screen(
            gpu_layers=args.gpu_layers, candidate_count=args.candidate_count,
            runs=args.runs, tokens=args.tokens, timeout=args.timeout,
        )
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
