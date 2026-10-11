"""Compare compound tensor placements motivated by the 40-layer screen.

This is a bounded interaction experiment, not an optimizer or a saved profile.
Baseline plus three *disjoint-rule compositions* are repeated in alternating
order; the raw per-run differences remain available to inspect noise.
"""

import argparse
import json
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark
from .mapping_candidates import propose_mapping_candidates
from .mapping_composition import compose_candidates
from .mapping_screen import choose_exploration_cohort


def choose_compound_trials(proposals: dict) -> list[dict]:
    """Use the existing FFN/attention pilot as an evidence-motivated cohort.

    Designs: promoting both small weights; exchanging attention evictions
    for FFN promotion; reclaiming FFN space and promoting two weights.
    Neither formula nor GGUF size claims to predict runtime throughput.
    """
    cohort = choose_exploration_cohort(proposals, 6)
    index = {(c["family"], c["direction"]): c for c in cohort}
    needed = [
        ("feed_forward", "host_to_gpu"),
        ("attention", "host_to_gpu"),
        ("attention", "gpu_to_host"),
        ("feed_forward", "gpu_to_host"),
    ]
    if any(key not in index for key in needed):
        raise ValueError("Offload iniziale non adatto a questi esperimenti combinati.")

    ffn_promote = index[("feed_forward", "host_to_gpu")]["id"]
    attn_promote = index[("attention", "host_to_gpu")]["id"]
    attn_evict = index[("attention", "gpu_to_host")]["id"]
    ffn_evict = index[("feed_forward", "gpu_to_host")]["id"]

    trials = [
        {"id": "baseline", "component_ids": [], "hypothesis": "Unmodified GPU offload"},
        {"id": "dual_promotion",
         "component_ids": [ffn_promote, attn_promote],
         "hypothesis": "Combine two small host-to-GPU promotions"},
        {"id": "cross_family_exchange",
         "component_ids": [attn_evict, ffn_promote],
         "hypothesis": "Trade an attention group for one FFN promotion"},
        {"id": "reclaimed_ffn_capacity",
         "component_ids": [ffn_evict, ffn_promote, attn_promote],
         "hypothesis": "Evict a GPU FFN group and promote two host tensors"},
    ]
    for entry in trials[1:]:
        composite = compose_candidates(proposals, entry["component_ids"])
        entry["tensor_names"] = composite["tensor_names"]
        entry["estimated_gguf_vram_delta_bytes"] = composite[
            "estimated_vram_weight_delta_bytes"
        ]
    return trials


def summarize_compound_trials(trials: list[dict], definitions: list[dict],
                              runs: int) -> list[dict]:
    by_id = {c["id"]: c for c in definitions}
    baseline = {t["run"]: t for t in trials if t["trial_id"] == "baseline"}
    if set(baseline) != set(range(1, runs + 1)):
        raise ValueError("Baseline incompleta o ripetizioni non valide.")
    result = []
    for id_, definition in by_id.items():
        series = [t for t in trials if t["trial_id"] == id_]
        if len(series) != runs or {t["run"] for t in series} != set(range(1, runs + 1)):
            raise ValueError(f"Ripetizioni incomplete o duplicate: {id_}")
        if any(not t["success"] or not all(
            isinstance(t.get(m), (int, float)) and t[m] > 0
            for m in ("decode_tps", "prefill_tps", "vram_peak_mib")
        ) for t in series):
            raise ValueError(f"Campione incompleto: {id_}")

        paired_decode_deltas = [
            round(t["decode_tps"] - baseline[t["run"]]["decode_tps"], 3)
            for t in series
        ]
        paired_prefill_deltas = [
            round(t["prefill_tps"] - baseline[t["run"]]["prefill_tps"], 3)
            for t in series
        ]
        result.append({
            "trial_id": id_,
            "hypothesis": definition["hypothesis"],
            "median_decode_tps": round(statistics.median(t["decode_tps"] for t in series), 2),
            "decode_range_tps": [
                min(t["decode_tps"] for t in series),
                max(t["decode_tps"] for t in series),
            ],
            "median_prefill_tps": round(statistics.median(t["prefill_tps"] for t in series), 2),
            "prefill_range_tps": [
                min(t["prefill_tps"] for t in series),
                max(t["prefill_tps"] for t in series),
            ],
            "max_sampled_vram_mib": max(t["vram_peak_mib"] for t in series),
            "paired_decode_delta_tps": paired_decode_deltas,
            "paired_prefill_delta_tps": paired_prefill_deltas,
            "decode_faster_in_every_pair": (
                id_ != "baseline" and all(d > 0 for d in paired_decode_deltas)
            ),
        })
    return result


def compare(gpu_layers: int = 40, runs: int = 3,
            tokens: int = 128, timeout: int = 180) -> dict:
    if not 2 <= runs <= 5 or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("runs=2..5; tokens=64..256; timeout>=1.")

    state = GPUManager.calibration_status()
    if not state["ready"]:
        raise RuntimeError(f"GPU non pronta: {state['errors']}; {state['foreign_processes']}")

    proposals = propose_mapping_candidates(
        ModelManager().inspect("gemma-2-9b-it", n_ctx=512), gpu_layers
    )
    definitions = choose_compound_trials(proposals)
    report = {
        "phase": "granular_mapping_compound_validation",
        "model": "gemma-2-9b-it",
        "backend": "cuda",
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "repetitions": runs,
        "tokens": tokens,
        "tested_compositions": definitions,
        "trials": [],
        "summary": [],
        "success": False,
        "profile_generated": False,
        "note": (
            "Esperimento sulle interazioni dei tensori, non scelta di mapping ottimo. "
            "Ogni prova e' isolata e verifica tutte le regole native. "
            "Delta per coppia e intervalli sono descrittivi, non intervalli "
            "di confidenza. Picchi VRAM NVML campionati."
        ),
    }

    for round_number in range(1, runs + 1):
        order = definitions if round_number % 2 == 1 else definitions[::-1]
        for entry in order:
            current = GPUManager.calibration_status()
            if not current["ready"]:
                report["error"] = (
                    f"GPU modificata: {current['errors']}; "
                    f"{current['foreign_processes']}"
                )
                return report
            observed = benchmark(
                gpu_layers=gpu_layers, max_tokens=tokens,
                timeout=timeout,
                combination_ids=entry["component_ids"] or None,
            )
            measurements = observed.get("measurements") or {}
            verification = observed.get("mapping_verification")
            valid = (
                observed.get("success") is True
                and (not entry["component_ids"] or
                     isinstance(verification, dict) and verification.get("verified") is True)
            )
            trial = {
                "run": round_number,
                "trial_id": entry["id"],
                "success": valid,
                "decode_tps": measurements.get("decode_tokens_per_second"),
                "prefill_tps": measurements.get("prefill_tokens_per_second"),
                "vram_peak_mib": measurements.get("vram_peak_sampled_mib"),
                "generated_tokens": measurements.get("generated_tokens"),
                "mapping_verification": verification,
                "worker_returncode": observed.get("worker_returncode"),
                "failure_type": observed.get("failure_type"),
                "native_signal": observed.get("native_signal"),
                "error": observed.get("error"),
            }
            report["trials"].append(trial)
            if not valid:
                report["error"] = (
                    f"Esperimento {entry['id']} #{round_number} interrotto: "
                    f"{trial['error'] or trial['failure_type'] or 'log nativi non confermati'}"
                )
                return report

    report["summary"] = summarize_compound_trials(report["trials"], definitions, runs)
    report["success"] = True
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Confronto di composizioni FFN/attention ad alto offload GPU"
    )
    parser.add_argument("--gpu-layers", type=int, default=40)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        result = compare(args.gpu_layers, args.runs, args.tokens, args.timeout)
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
