"""Counterbalanced A/B performance study with NVML stability telemetry.

ABBA and BAAB batches reduce linear time-drift bias in two-way comparisons.
The model and mappings are identical to earlier native-verified diagnostics.
n_ctx=512 remains a probe only; no output or production profile is selected.
"""

import argparse
import json
import math
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark, validate_probe
from .mapping_candidates import propose_mapping_candidates
from .mapping_composition import compose_candidates
from .measured_combinations import choose_compound_trials
from .window_stability import compare_decode_windows


def schedule(rounds: int) -> list[list[str]]:
    if not isinstance(rounds, int) or not 2 <= rounds <= 8:
        raise ValueError("rounds deve essere compreso fra 2 e 8.")
    return [
        ["baseline", "candidate", "candidate", "baseline"]
        if run % 2 else ["candidate", "baseline", "baseline", "candidate"]
        for run in range(1, rounds + 1)
    ]


def summarize_stability(trials: list[dict], rounds: int, gpu_total_mib: float) -> dict:
    if not isinstance(gpu_total_mib, (int, float)) or gpu_total_mib <= 0:
        raise ValueError("GPU totale non disponibile.")
    plan = schedule(rounds)
    by_key = {}
    for trial in trials:
        batch = trial.get("round")
        order = trial.get("position")
        variant = trial.get("variant")
        if (not isinstance(batch, int) or batch not in range(1, rounds + 1)
            or not isinstance(order, int) or order not in (1, 2, 3, 4)
            or variant != plan[batch - 1][order - 1]
            or (batch, order) in by_key):
            raise ValueError("Esperimento sconosciuto, duplicato o ordine errato.")
        if trial.get("success") is not True:
            raise ValueError("Non aggregare prove fallite.")
        if variant == "candidate" and trial.get("mapping_verified") is not True:
            raise ValueError("Override nativo non verificato.")
        if trial.get("generated_tokens") != trial.get("requested_tokens"):
            raise ValueError("Generazione interrotta prima del numero richiesto di token.")
        for key in ("decode_tps", "prefill_tps", "vram_peak_mib"):
            number = trial.get(key)
            if (not isinstance(number, (int, float)) or isinstance(number, bool)
                or not math.isfinite(number) or number <= 0):
                raise ValueError(f"Misura numerica assente: {key}")
        by_key[batch, order] = trial

    if len(by_key) != rounds * 4:
        raise ValueError("Campionamento incompleto.")

    batches = []
    for batch in range(1, rounds + 1):
        rows = [by_key[batch, position] for position in (1, 2, 3, 4)]
        baseline = [r for r in rows if r["variant"] == "baseline"]
        candidate = [r for r in rows if r["variant"] == "candidate"]
        def avg(seq, key):
            return statistics.mean(r[key] for r in seq)
        base_decode = avg(baseline, "decode_tps")
        candidate_decode = avg(candidate, "decode_tps")
        base_prefill = avg(baseline, "prefill_tps")
        candidate_prefill = avg(candidate, "prefill_tps")
        batches.append({
            "round": batch,
            "sequence": plan[batch - 1],
            "baseline_decode_tps_mean": round(base_decode, 3),
            "candidate_decode_tps_mean": round(candidate_decode, 3),
            "decode_delta_tps": round(candidate_decode - base_decode, 3),
            "decode_delta_percent": round(100 * (candidate_decode / base_decode - 1), 3),
            "baseline_prefill_tps_mean": round(base_prefill, 3),
            "candidate_prefill_tps_mean": round(candidate_prefill, 3),
            "prefill_delta_tps": round(candidate_prefill - base_prefill, 3),
            "vram_peak_delta_mib": round(
                max(r["vram_peak_mib"] for r in candidate)
                - max(r["vram_peak_mib"] for r in baseline), 1
            ),
            "decode_clock_median_by_variant_mhz": {
                variant: statistics.median(
                    row["gpu_decode_sm_clock_median_mhz"]
                    for row in rows if row["variant"] == variant
                    and isinstance(row.get("gpu_decode_sm_clock_median_mhz"), (int, float))
                ) if any(
                    isinstance(row.get("gpu_decode_sm_clock_median_mhz"), (int, float))
                    for row in rows if row["variant"] == variant
                ) else None for variant in ("baseline", "candidate")
            },
            "decode_temperature_median_by_variant_c": {
                variant: statistics.median(
                    row["gpu_decode_temperature_median_c"]
                    for row in rows if row["variant"] == variant
                    and isinstance(row.get("gpu_decode_temperature_median_c"), (int, float))
                ) if any(
                    isinstance(row.get("gpu_decode_temperature_median_c"), (int, float))
                    for row in rows if row["variant"] == variant
                ) else None for variant in ("baseline", "candidate")
            },
        })

    def median_metric(variant, metric):
        return round(statistics.median(
            row[metric] for row in trials if row["variant"] == variant
        ), 2)

    decode = [r["decode_delta_tps"] for r in batches]
    prefill = [r["prefill_delta_tps"] for r in batches]
    vram = [r["vram_peak_mib"] for r in trials if r["variant"] == "candidate"]
    segmented = compare_decode_windows(trials, rounds)
    return {
        "baseline": {
            "median_decode_tps": median_metric("baseline", "decode_tps"),
            "median_prefill_tps": median_metric("baseline", "prefill_tps"),
        },
        "candidate": {
            "median_decode_tps": median_metric("candidate", "decode_tps"),
            "median_prefill_tps": median_metric("candidate", "prefill_tps"),
        },
        "rounds": batches,
        "counterbalanced_decode_windows": segmented,
        "median_decode_delta_tps": round(statistics.median(decode), 3),
        "median_prefill_delta_tps": round(statistics.median(prefill), 3),
        "decode_positive_rounds": sum(x > 0 for x in decode),
        "prefill_positive_rounds": sum(x > 0 for x in prefill),
        "decode_delta_range_tps": [min(decode), max(decode)],
        "prefill_delta_range_tps": [min(prefill), max(prefill)],
        "max_sampled_vram_mib_candidate": max(vram),
        "sampled_candidate_headroom_mib": round(gpu_total_mib - max(vram), 1),
        "descriptive_only": True,
        "candidate_selected_for_runtime": False,
        "note": (
            "L'ordine ABBA/BAAB compensa soltanto una deriva lineare ideale. "
            "Frequenze e temperature rilevate non dimostrano la causa del rumore. "
            "Queste prove sono diagnostiche e non scelgono un mapping operativo."
        ),
    }


def run_stability(gpu_layers: int = 40, rounds: int = 3,
                  tokens: int = 256, timeout: int = 180,
                  probe_n_ctx: int = 512, prompt_repeat: int = 1) -> dict:
    plan = schedule(rounds)
    validate_probe(probe_n_ctx, prompt_repeat, tokens)
    if not isinstance(gpu_layers, int) or gpu_layers < 1:
        raise ValueError("gpu_layers deve essere positivo.")
    if not isinstance(tokens, int) or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("tokens=64..256 e timeout>=1.")
    state = GPUManager.calibration_status()
    if not state["ready"]:
        raise RuntimeError(f"GPU non pronta: {state['errors']}; {state['foreign_processes']}")
    proposals = propose_mapping_candidates(
        ModelManager().inspect("gemma-2-9b-it", n_ctx=probe_n_ctx), gpu_layers
    )
    seed = next(
        c for c in choose_compound_trials(proposals) if c["id"] == "dual_promotion"
    )
    compose_candidates(proposals, seed["component_ids"])

    report = {
        "phase": "mapping_counterbalanced_gpu_stability",
        "model": "gemma-2-9b-it",
        "backend": "cuda",
        "gpu_layers": gpu_layers,
        "probe_n_ctx": probe_n_ctx,
        "prompt_repeat": prompt_repeat,
        "selected_n_ctx": None,
        "tokens": tokens,
        "rounds": rounds,
        "schedule": plan,
        "candidate": seed,
        "gpu_total_mib": (state.get("memory") or {}).get("total_mib"),
        "trials": [],
        "summary": None,
        "success": False,
        "profile_generated": False,
        "note": (
            "Esperimento controbilanciato ABBA/BAAB con telemetria GPU. "
            "Le velocita' di quattro quartili sono confrontate in modo "
            "controbilanciato fra le due varianti. Ogni prova usa un worker isolato. "
            "Nessuna decisione su mapping o n_ctx operativo."
        ),
    }
    for round_idx, order in enumerate(plan, start=1):
        for position, variant in enumerate(order, start=1):
            current = GPUManager.calibration_status()
            if not current["ready"]:
                report["error"] = f"GPU cambiata: {current['errors']}; {current['foreign_processes']}"
                return report
            outcome = benchmark(
                gpu_layers=gpu_layers, max_tokens=tokens, timeout=timeout,
                combination_ids=seed["component_ids"] if variant == "candidate" else None,
                collect_gpu_telemetry=True,
                probe_n_ctx=probe_n_ctx,
                prompt_repeat=prompt_repeat,
            )
            metrics = outcome.get("measurements") or {}
            decode_windows = metrics.get("decode_windows") or {}
            checks = outcome.get("mapping_verification") or {}
            telemetry = (outcome.get("gpu_telemetry_by_phase") or {}).get("decode") or {}
            temp = telemetry.get("gpu_temperature_c") or {}
            clock = telemetry.get("sm_clock_mhz") or {}
            row = {
                "round": round_idx,
                "position": position,
                "variant": variant,
                "success": outcome.get("success") is True,
                "mapping_verified": (
                    checks.get("verified") is True if variant == "candidate" else None
                ),
                "requested_tokens": tokens,
                "actual_probe_n_ctx": outcome.get("probe_n_ctx"),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "generated_tokens": metrics.get("generated_tokens"),
                "decode_window_details": decode_windows.get("windows"),
                "final_context_token_fraction": decode_windows.get("end_context_fraction"),
                "decode_tps": metrics.get("decode_tokens_per_second"),
                "prefill_tps": metrics.get("prefill_tokens_per_second"),
                "vram_peak_mib": metrics.get("vram_peak_sampled_mib"),
                "gpu_decode_temperature_median_c": temp.get("median"),
                "gpu_decode_sm_clock_median_mhz": clock.get("median"),
                "gpu_decode_telemetry_samples": telemetry.get("sample_count"),
                "worker_returncode": outcome.get("worker_returncode"),
                "failure_type": outcome.get("failure_type"),
                "error": outcome.get("error"),
            }
            report["trials"].append(row)
            if (not row["success"]
                or (variant == "candidate" and not row["mapping_verified"])
                or row["generated_tokens"] != tokens
                or row["actual_probe_n_ctx"] != probe_n_ctx
                or not isinstance(row["decode_window_details"], list)
                or len(row["decode_window_details"]) != 4):
                report["error"] = (
                    f"Misura incompleta: round {round_idx} slot {position} "
                    f"{variant}, {row['error'] or row['failure_type'] or 'token insufficienti'}"
                )
                return report
    report["summary"] = summarize_stability(
        report["trials"], rounds, report["gpu_total_mib"]
    )
    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronto CUDA ABBA/BAAB del mapping sotto deriva di performance"
    )
    parser.add_argument("--gpu-layers", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--probe-n-ctx", type=int, default=512,
                        help="Solo contesto diagnostico, non profilo operativo")
    parser.add_argument("--prompt-repeat", type=int, default=1,
                        help="Ripetizioni del testo del prompt diagnostico")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        result = run_stability(
            args.gpu_layers, args.rounds, args.tokens, args.timeout,
            args.probe_n_ctx, args.prompt_repeat,
        )
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
