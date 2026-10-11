"""Measure actual costs of both directions of granular CUDA placement.

Compare the native-validated small attention tensor candidates against the
unmodified baseline, at the same technical n_ctx=512. This is NOT a profile
optimizer; in particular, no context or quality decision is made here.
"""

import argparse
import json
import statistics
import sys

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark
from .mapping_candidates import propose_mapping_candidates
from .mapping_native_validation import choose_native_trials


def summarize_trials(trials: list[dict], candidates: list[dict], runs: int) -> list[dict]:
    """Summarize complete, verified measurements; no artificial score/weights."""
    summaries = []
    for candidate in candidates:
        rows = [row for row in trials if row["candidate_id"] == candidate["id"]]
        if len(rows) != runs or any(not row["success"] for row in rows):
            raise ValueError(f"Prove mancanti o fallite per {candidate['id']}.")

        columns = (
            "decode_tps", "prefill_tps", "vram_peak_mib",
        )
        if any(
            not isinstance(row.get(metric), (int, float)) or row[metric] <= 0
            for row in rows for metric in columns
        ):
            raise ValueError(f"Misure invalide per {candidate['id']}.")
        if any(
            row.get("generated_tokens", 0) < 48
            for row in rows
        ):
            raise ValueError(f"Generazione troppo breve per {candidate['id']}.")

        summary = {
            "candidate_id": candidate["id"],
            "direction": candidate["direction"],
            "family": candidate["family"],
            "median_decode_tps": round(statistics.median(r["decode_tps"] for r in rows), 2),
            "median_prefill_tps": round(statistics.median(r["prefill_tps"] for r in rows), 2),
            "max_sampled_vram_peak_mib": max(r["vram_peak_mib"] for r in rows),
            "median_vram_peak_mib": round(
                statistics.median(r["vram_peak_mib"] for r in rows), 1
            ),
        }
        summaries.append(summary)

    baseline = next(row for row in summaries if row["candidate_id"] == "baseline")
    for item in summaries:
        item["decode_change_percent_vs_baseline"] = round(
            100 * (item["median_decode_tps"] / baseline["median_decode_tps"] - 1), 2
        )
        item["prefill_change_percent_vs_baseline"] = round(
            100 * (item["median_prefill_tps"] / baseline["median_prefill_tps"] - 1), 2
        )
        item["sampled_vram_peak_change_mib_vs_baseline"] = round(
            item["max_sampled_vram_peak_mib"] -
            baseline["max_sampled_vram_peak_mib"], 1
        )
    return summaries


def compare(gpu_layers: int = 32, runs: int = 2,
            tokens: int = 128, timeout: int = 180) -> dict:
    if not 1 <= runs <= 5 or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("runs=1..5; tokens=64..256; timeout>=1.")
    before = GPUManager.calibration_status()
    if not before["ready"]:
        raise RuntimeError(
            f"GPU non pronta: {before['errors']}; "
            f"processi esterni={before['foreign_processes']}"
        )

    layout = ModelManager().inspect("gemma-2-9b-it", n_ctx=512)
    proposals = propose_mapping_candidates(layout, gpu_layers)
    chosen = choose_native_trials(proposals)
    baseline = next(c for c in proposals["candidates"] if c["id"] == "baseline")
    candidates = [baseline] + chosen

    report = {
        "phase": "bidirectional_mapping_performance_comparison",
        "model": "gemma-2-9b-it",
        "engine": "llama.cpp",
        "method": "llama-cpp-python",
        "gpu": before["gpu"]["name"],
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "generation_test_tokens": tokens,
        "runs_per_candidate": runs,
        "total_candidate_count": proposals["candidate_count"],
        "benchmarked_candidate_count": len(candidates),
        "candidates": [
            {"id": c["id"], "direction": c["direction"],
             "family": c["family"], "rules": c["rules"],
             "estimated_vram_weight_delta_bytes": c["estimated_vram_weight_delta_bytes"]}
            for c in candidates
        ],
        "trials": [],
        "summaries": [],
        "success": False,
        "profile_generated": False,
        "note": (
            "Confronto diagnostico alternato della baseline e tre mapping granulari. "
            "La verifica dei buffer nativi e' obbligatoria per ogni prova non-baseline. "
            "Nessuna classifica definitiva, quality test o scelta di n_ctx. "
            "Il picco NVML e' campionato, non garantito assoluto."
        ),
    }

    for cycle in range(runs):
        # Reverse even cycles to reduce systematic order/temperature bias.
        order = candidates if cycle % 2 == 0 else list(reversed(candidates))
        for c in order:
            state = GPUManager.calibration_status()
            if not state["ready"]:
                report["error"] = (
                    f"Ambiente GPU modificato: {state['errors']}; "
                    f"processi esterni={state['foreign_processes']}"
                )
                return report
            result = benchmark(
                gpu_layers=gpu_layers,
                max_tokens=tokens,
                timeout=timeout,
                candidate_id=c["id"] if c["rules"] else None,
            )
            measurements = result.get("measurements") or {}
            trial = {
                "run": cycle + 1,
                "candidate_id": c["id"],
                "direction": c["direction"],
                "success": result.get("success") is True,
                "decode_tps": measurements.get("decode_tokens_per_second"),
                "prefill_tps": measurements.get("prefill_tokens_per_second"),
                "vram_peak_mib": measurements.get("vram_peak_sampled_mib"),
                "generated_tokens": measurements.get("generated_tokens"),
                "prompt_tokens": measurements.get("prompt_tokens"),
                "mapping_verification": result.get("mapping_verification"),
                "worker_returncode": result.get("worker_returncode"),
                "failure_type": result.get("failure_type"),
                "native_signal": result.get("native_signal"),
                "error": result.get("error"),
            }
            report["trials"].append(trial)
            if not trial["success"]:
                report["error"] = (
                    f"Benchmark {c['id']} (prova {cycle+1}) non valido: "
                    f"{trial['error'] or trial['failure_type']}"
                )
                return report

    report["summaries"] = summarize_trials(report["trials"], candidates, runs)
    report["success"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronto misurato di baseline, host, GPU e swap nativi"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2)
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
