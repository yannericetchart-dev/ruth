"""Ablation study of an empirically promising tensor placement.

Compare the native default baseline, a full seed mapping and every
leave-one-component-out variant. All placements derive from the GGUF
candidate generator. Results are paired within each repeated round,
without generating an operational profile or setting an operational n_ctx.
"""

import argparse
import json
import math
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark
from .mapping_candidates import propose_mapping_candidates
from .mapping_composition import compose_candidates
from .measured_combinations import choose_compound_trials


def build_ablation_variants(proposals: dict, component_ids: list[str]) -> list[dict]:
    """Build one full-seed, one empty, and every leave-one-out placement.

    The generator supports arbitrary GGUF tensor families and exact native
    rules. Ablations are explicit: absence of an override restores the
    llama.cpp default for that component rather than forcing it elsewhere.
    """
    if not 2 <= len(component_ids) <= 6:
        raise ValueError("Ablazione supportata per 2..6 componenti.")
    full = compose_candidates(proposals, component_ids)
    variants = [
        {
            "id": "baseline",
            "component_ids": [],
            "removed_component": None,
            "tensor_names": [],
            "kind": "native_default",
        },
        {
            "id": "full_seed",
            "component_ids": list(component_ids),
            "removed_component": None,
            "tensor_names": full["tensor_names"],
            "kind": "reference_seed",
        },
    ]
    for removed_index, removed in enumerate(component_ids):
        subset = [
            cid for index, cid in enumerate(component_ids)
            if index != removed_index
        ]
        composed = compose_candidates(proposals, subset)
        variants.append({
            "id": f"without_{removed_index}",
            "component_ids": subset,
            "removed_component": removed,
            "tensor_names": composed["tensor_names"],
            "kind": "leave_one_out",
        })
    return variants


def summarize_ablation(trials: list[dict], variants: list[dict],
                       runs: int, gpu_total_mib: float) -> dict:
    """Summarize measured contributions and reversibility, not causality claims."""
    if not isinstance(runs, int) or runs < 2 or gpu_total_mib <= 0:
        raise ValueError("Dimensioni esperimento non valide.")
    ids = [v["id"] for v in variants]
    if len(ids) != len(set(ids)) or "baseline" not in ids or "full_seed" not in ids:
        raise ValueError("Definizioni varianti incomplete.")
    by_key = {}
    for row in trials:
        key = (row.get("run"), row.get("variant_id"))
        if (not isinstance(key[0], int) or key[0] not in range(1, runs + 1)
            or key[1] not in ids or key in by_key):
            raise ValueError("Misura duplicata o sconosciuta.")
        if row.get("success") is not True:
            raise ValueError("Sono presenti misurazioni fallite.")
        if row["variant_id"] != "baseline" and row.get("native_verified") is not True:
            raise ValueError("Override nativi non verificati.")
        if not isinstance(row.get("generated_tokens"), int) or row["generated_tokens"] < 48:
            raise ValueError("Numero di token di decodifica insufficiente.")
        for metric in ("decode_tps", "prefill_tps", "vram_peak_mib"):
            value = row.get(metric)
            if (not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value) or value <= 0):
                raise ValueError(f"Metrica mancante/non valida: {metric}")
        by_key[key] = row

    if len(by_key) != len(variants) * runs:
        raise ValueError("Ripetizioni incomplete.")

    def rows_for(variant_id):
        return [by_key[(run, variant_id)] for run in range(1, runs + 1)]

    def diffs(candidate, reference, metric):
        return [
            round(by_key[(run, candidate)][metric]
                  - by_key[(run, reference)][metric], 3)
            for run in range(1, runs + 1)
        ]

    outcomes = []
    for variant in variants:
        id_ = variant["id"]
        rows = rows_for(id_)
        peak = max(r["vram_peak_mib"] for r in rows)
        decode_vs_baseline = diffs(id_, "baseline", "decode_tps")
        decode_vs_seed = diffs(id_, "full_seed", "decode_tps")
        prefill_vs_baseline = diffs(id_, "baseline", "prefill_tps")
        prefill_vs_seed = diffs(id_, "full_seed", "prefill_tps")
        vram_vs_seed = diffs(id_, "full_seed", "vram_peak_mib")
        outcomes.append({
            "id": id_,
            "kind": variant["kind"],
            "component_ids": variant["component_ids"],
            "removed_component": variant["removed_component"],
            "median_decode_tps": round(
                statistics.median(r["decode_tps"] for r in rows), 2
            ),
            "median_prefill_tps": round(
                statistics.median(r["prefill_tps"] for r in rows), 2
            ),
            "max_sampled_vram_peak_mib": peak,
            "sampled_headroom_at_peak_mib": round(gpu_total_mib - peak, 1),
            "paired_decode_vs_baseline_tps": decode_vs_baseline,
            "paired_decode_vs_full_seed_tps": decode_vs_seed,
            "paired_prefill_vs_baseline_tps": prefill_vs_baseline,
            "paired_prefill_vs_full_seed_tps": prefill_vs_seed,
            "paired_vram_vs_full_seed_mib": vram_vs_seed,
            "decode_faster_than_seed_pairs": sum(x > 0 for x in decode_vs_seed),
            "decode_faster_than_baseline_pairs": sum(x > 0 for x in decode_vs_baseline),
        })

    # The marginal contribution of a component means full seed minus the
    # mapping with that component removed; it may reflect interactions.
    contributions = []
    for variant in variants:
        if variant["kind"] != "leave_one_out":
            continue
        without = variant["id"]
        decode = diffs("full_seed", without, "decode_tps")
        prefill = diffs("full_seed", without, "prefill_tps")
        memory = diffs("full_seed", without, "vram_peak_mib")
        contributions.append({
            "component_id": variant["removed_component"],
            "leave_one_out_id": without,
            "marginal_decode_tps_by_round": decode,
            "marginal_prefill_tps_by_round": prefill,
            "marginal_vram_mib_by_round": memory,
            "median_marginal_decode_tps": round(statistics.median(decode), 3),
            "full_seed_decode_faster_pairs": sum(x > 0 for x in decode),
            "full_seed_prefill_faster_pairs": sum(x > 0 for x in prefill),
            "interpretation": "Contributo condizionale: non isola interazioni tra tensori.",
        })

    interaction = None
    leave_one_out_ids = [
        variant["id"] for variant in variants
        if variant["kind"] == "leave_one_out"
    ]
    if len(leave_one_out_ids) == 2:
        # For two components, leave-one-out variants are the individual
        # mappings. This describes whether the pair is additive in decode;
        # it is not proof of a causal synergy or statistical significance.
        changes = [
            round(
                by_key[(run, "full_seed")]["decode_tps"]
                - sum(by_key[(run, id_)]["decode_tps"]
                      for id_ in leave_one_out_ids)
                + by_key[(run, "baseline")]["decode_tps"],
                3,
            )
            for run in range(1, runs + 1)
        ]
        interaction = {
            "decode_nonadditivity_by_round_tps": changes,
            "median_decode_nonadditivity_tps": round(statistics.median(changes), 3),
            "positive_rounds": sum(v > 0 for v in changes),
            "note": (
                "Interazione descrittiva pair - singleA - singleB + default, "
                "sensibile alla variabilita' temporale."
            ),
        }

    return {
        "variants": outcomes,
        "component_marginal_contributions": contributions,
        "two_component_interaction": interaction,
        "descriptive_only": True,
        "conclusion": (
            "Confrontare differenze per coppia e allocazione VRAM. "
            "Nessun mapping viene approvato automaticamente."
        ),
    }


def run_ablation(gpu_layers: int = 40, runs: int = 5, tokens: int = 128,
                 timeout: int = 180, seed_ids: list[str] | None = None) -> dict:
    if (not isinstance(gpu_layers, int) or gpu_layers < 1
        or not isinstance(runs, int) or not 2 <= runs <= 8
        or not isinstance(tokens, int) or not 64 <= tokens <= 256
        or timeout < 1):
        raise ValueError("gpu_layers>=1, runs=2..8, tokens=64..256, timeout>=1.")

    state = GPUManager.calibration_status()
    if not state["ready"]:
        raise RuntimeError(f"GPU non pronta: {state['errors']}; {state['foreign_processes']}")
    gpu_total_mib = (state.get("memory") or {}).get("total_mib")
    if not isinstance(gpu_total_mib, (int, float)) or gpu_total_mib <= 0:
        raise RuntimeError("Capacita' VRAM NVML non disponibile.")
    proposals = propose_mapping_candidates(
        ModelManager().inspect("gemma-2-9b-it", n_ctx=512), gpu_layers
    )
    if seed_ids is None:
        source = choose_compound_trials(proposals)
        seed_ids = next(
            item["component_ids"] for item in source
            if item["id"] == "dual_promotion"
        )
    variants = build_ablation_variants(proposals, seed_ids)

    report = {
        "phase": "tensor_mapping_component_ablation",
        "model": "gemma-2-9b-it",
        "backend": "cuda",
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "runs_per_variant": runs,
        "generation_tokens": tokens,
        "gpu_total_mib": gpu_total_mib,
        "seed_component_ids": list(seed_ids),
        "variants": variants,
        "trials": [],
        "comparison": None,
        "success": False,
        "profile_generated": False,
        "note": (
            "Ablazione del mapping derivato da misure precedenti. "
            "Tutte le varianti su processi isolati con ordine inverso alternato. "
            "I confronti sono descrittivi: la calibrazione del contesto, "
            "qualita' e un profilo finale NON sono parte di questo test."
        ),
    }
    for run in range(1, runs + 1):
        order = variants if run % 2 else list(reversed(variants))
        for variant in order:
            current = GPUManager.calibration_status()
            if not current["ready"]:
                report["error"] = (
                    f"GPU mutata durante ablazione: {current['errors']}; "
                    f"{current['foreign_processes']}"
                )
                return report
            observed = benchmark(
                gpu_layers=gpu_layers, max_tokens=tokens, timeout=timeout,
                combination_ids=variant["component_ids"] or None,
            )
            measurements = observed.get("measurements") or {}
            verification = observed.get("mapping_verification") or {}
            verified = (
                verification.get("verified") is True
                if variant["component_ids"] else None
            )
            valid = observed.get("success") is True and (
                not variant["component_ids"] or verified is True
            )
            row = {
                "run": run,
                "variant_id": variant["id"],
                "success": valid,
                "native_verified": verified,
                "decode_tps": measurements.get("decode_tokens_per_second"),
                "prefill_tps": measurements.get("prefill_tokens_per_second"),
                "vram_peak_mib": measurements.get("vram_peak_sampled_mib"),
                "generated_tokens": measurements.get("generated_tokens"),
                "worker_returncode": observed.get("worker_returncode"),
                "failure_type": observed.get("failure_type"),
                "native_signal": observed.get("native_signal"),
                "error": observed.get("error"),
            }
            report["trials"].append(row)
            if not valid:
                report["error"] = (
                    f"Test interrotto: {variant['id']} round {run}, "
                    f"{row['error'] or row['failure_type'] or 'override non verificato'}"
                )
                return report

    report["comparison"] = summarize_ablation(
        report["trials"], variants, runs, gpu_total_mib
    )
    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablazione dei tensori di un mapping granulare confermato"
    )
    parser.add_argument("--gpu-layers", type=int, default=40)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--seed-component", action="append", default=None,
                        help="ID GGUF da includere nel seed; ripeti per componenti diversi")
    args = parser.parse_args()
    try:
        report = run_ablation(
            gpu_layers=args.gpu_layers, runs=args.runs,
            tokens=args.tokens, timeout=args.timeout,
            seed_ids=args.seed_component,
        )
    except Exception as exc:
        report = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
