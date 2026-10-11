"""Focused, paired confirmation of an observed high-offload tensor mapping.

Two variants (unmodified baseline versus one previously screened composite)
are remeasured at the *same technical* n_ctx=512, with reversed ordering
across successive rounds. This is a diagnostic repetition and never creates
or selects a production profile.
"""

import argparse
import json
import math
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark
from .mapping_candidates import propose_mapping_candidates
from .measured_combinations import choose_compound_trials


def summarize_confirmation(trials: list[dict], rounds: int) -> dict:
    """Report paired evidence without pretending to know its noise distribution."""
    if not isinstance(rounds, int) or rounds < 2:
        raise ValueError("Servono almeno due coppie.")
    pairs = {}
    for trial in trials:
        run, variant = trial.get("run"), trial.get("variant")
        if (not isinstance(run, int) or not 1 <= run <= rounds
            or variant not in ("baseline", "candidate")
            or (run, variant) in pairs):
            raise ValueError("Prova duplicata o con identificatore non valido.")
        if trial.get("success") is not True:
            raise ValueError("Non usare prove native fallite.")
        required = ("decode_tps", "prefill_tps", "vram_peak_mib")
        if any(
            not isinstance(trial.get(key), (int, float))
            or not math.isfinite(trial[key]) or trial[key] <= 0
            for key in required
        ):
            raise ValueError("Misure prestazionali mancanti o non valide.")
        if not isinstance(trial.get("generated_tokens"), int) or trial["generated_tokens"] < 48:
            raise ValueError("Campione di decodifica insufficiente.")
        if variant == "candidate" and trial.get("native_mapping_verified") is not True:
            raise ValueError("Override nativi non confermati.")
        pairs[(run, variant)] = trial

    if len(pairs) != 2 * rounds:
        raise ValueError("Non tutte le coppie baseline/candidato sono presenti.")

    detailed = []
    for run in range(1, rounds + 1):
        baseline = pairs[(run, "baseline")]
        candidate = pairs[(run, "candidate")]
        detailed.append({
            "run": run,
            "baseline_decode_tps": baseline["decode_tps"],
            "candidate_decode_tps": candidate["decode_tps"],
            "decode_delta_tps": round(candidate["decode_tps"] - baseline["decode_tps"], 3),
            "decode_delta_percent": round(
                100 * (candidate["decode_tps"] / baseline["decode_tps"] - 1), 3
            ),
            "baseline_prefill_tps": baseline["prefill_tps"],
            "candidate_prefill_tps": candidate["prefill_tps"],
            "prefill_delta_tps": round(candidate["prefill_tps"] - baseline["prefill_tps"], 3),
            "prefill_delta_percent": round(
                100 * (candidate["prefill_tps"] / baseline["prefill_tps"] - 1), 3
            ),
            "baseline_vram_peak_mib": baseline["vram_peak_mib"],
            "candidate_vram_peak_mib": candidate["vram_peak_mib"],
            "vram_peak_delta_mib": round(
                candidate["vram_peak_mib"] - baseline["vram_peak_mib"], 1
            ),
        })

    def metric(variant, key):
        return [pairs[(run, variant)][key] for run in range(1, rounds + 1)]

    decode_changes = [r["decode_delta_tps"] for r in detailed]
    prefill_changes = [r["prefill_delta_tps"] for r in detailed]
    return {
        "baseline": {
            "median_decode_tps": round(statistics.median(metric("baseline", "decode_tps")), 2),
            "median_prefill_tps": round(statistics.median(metric("baseline", "prefill_tps")), 2),
            "max_sampled_vram_peak_mib": max(metric("baseline", "vram_peak_mib")),
        },
        "candidate": {
            "median_decode_tps": round(statistics.median(metric("candidate", "decode_tps")), 2),
            "median_prefill_tps": round(statistics.median(metric("candidate", "prefill_tps")), 2),
            "max_sampled_vram_peak_mib": max(metric("candidate", "vram_peak_mib")),
        },
        "paired_observations": detailed,
        "paired_decode_delta_median_tps": round(statistics.median(decode_changes), 3),
        "paired_prefill_delta_median_tps": round(statistics.median(prefill_changes), 3),
        "decode_improved_pairs": sum(d > 0 for d in decode_changes),
        "prefill_improved_pairs": sum(d > 0 for d in prefill_changes),
        "decode_improved_in_every_pair": all(d > 0 for d in decode_changes),
        "prefill_improved_in_every_pair": all(d > 0 for d in prefill_changes),
        "paired_decode_range_tps": [min(decode_changes), max(decode_changes)],
        "paired_prefill_range_tps": [min(prefill_changes), max(prefill_changes)],
        "descriptive_only": True,
        "note": (
            "Differenze abbinate e consistenza del segno non dimostrano "
            "significativita' statistica. Misure soggette a deriva termica, "
            "ordine, caching e campionamento NVML."
        ),
    }


def confirm(gpu_layers: int = 40, runs: int = 5, tokens: int = 128,
            trial_id: str = "dual_promotion", timeout: int = 180) -> dict:
    if not isinstance(gpu_layers, int) or gpu_layers < 1:
        raise ValueError("gpu_layers deve essere positivo.")
    if not isinstance(runs, int) or not 2 <= runs <= 10:
        raise ValueError("runs deve essere compreso fra 2 e 10.")
    if not isinstance(tokens, int) or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("tokens=64..256 e timeout>=1.")

    gpu = GPUManager.calibration_status()
    if not gpu["ready"]:
        raise RuntimeError(f"GPU non pronta: {gpu['errors']}; {gpu['foreign_processes']}")
    proposals = propose_mapping_candidates(
        ModelManager().inspect("gemma-2-9b-it", n_ctx=512), gpu_layers
    )
    definitions = choose_compound_trials(proposals)
    selected = next((c for c in definitions if c["id"] == trial_id), None)
    if selected is None or not selected["component_ids"]:
        raise ValueError(f"Composizione non valida o baseline: {trial_id}")

    report = {
        "phase": "mapping_candidate_paired_confirmation",
        "model": "gemma-2-9b-it",
        "backend": "cuda",
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "tokens": tokens,
        "repetitions_per_variant": runs,
        "candidate": selected,
        "gpu_total_mib_at_start": (gpu.get("memory") or {}).get("total_mib"),
        "trials": [],
        "comparison": None,
        "success": False,
        "profile_generated": False,
        "note": (
            "Conferma del candidato misurato, in worker isolati. "
            "Ordine alternato baseline/candidato per limitare i bias. "
            "Non assume un delta minimo significativo, non cambia il "
            "contesto e non genera un profilo. "
            "Memoria totale letta da NVML, picchi campionati."
        ),
    }

    for run in range(1, runs + 1):
        order = ("baseline", "candidate") if run % 2 else ("candidate", "baseline")
        for variant in order:
            state = GPUManager.calibration_status()
            if not state["ready"]:
                report["error"] = (
                    f"Stato GPU variato durante la conferma: "
                    f"{state['errors']}; {state['foreign_processes']}"
                )
                return report
            observed = benchmark(
                gpu_layers=gpu_layers,
                max_tokens=tokens, timeout=timeout,
                combination_ids=selected["component_ids"] if variant == "candidate" else None,
            )
            measurement = observed.get("measurements") or {}
            verify = observed.get("mapping_verification")
            native_valid = (
                isinstance(verify, dict) and verify.get("verified") is True
            ) if variant == "candidate" else None
            valid = observed.get("success") is True and (
                variant == "baseline" or native_valid is True
            )
            trial = {
                "run": run,
                "variant": variant,
                "success": valid,
                "native_mapping_verified": native_valid,
                "decode_tps": measurement.get("decode_tokens_per_second"),
                "prefill_tps": measurement.get("prefill_tokens_per_second"),
                "generated_tokens": measurement.get("generated_tokens"),
                "vram_peak_mib": measurement.get("vram_peak_sampled_mib"),
                "vram_loaded_mib": measurement.get("vram_used_loaded_mib"),
                "worker_returncode": observed.get("worker_returncode"),
                "failure_type": observed.get("failure_type"),
                "native_signal": observed.get("native_signal"),
                "error": observed.get("error"),
            }
            report["trials"].append(trial)
            if not valid:
                report["error"] = (
                    f"Conferma {variant} al giro {run} non valida: "
                    f"{trial['error'] or trial['failure_type'] or 'buffer nativi'}"
                )
                return report
    report["comparison"] = summarize_confirmation(report["trials"], runs)
    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conferma appaiata di mapping granulare a basso rumore"
    )
    parser.add_argument("--gpu-layers", type=int, default=40)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--trial", default="dual_promotion",
                        choices=("dual_promotion", "cross_family_exchange", "reclaimed_ffn_capacity"))
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        result = confirm(args.gpu_layers, args.runs, args.tokens, args.trial, args.timeout)
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
