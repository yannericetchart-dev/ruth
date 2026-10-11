"""Bounded, GGUF-driven exploratory screen of granular CUDA placements.

This is an experiment coordinator, not a profile optimizer. No fixed VRAM
reserve, quality score, or operational n_ctx is inferred from these trials.
"""

import argparse
from collections import defaultdict
import json
import statistics

from hardware import GPUManager
from model import ModelManager

from .benchmark import benchmark
from .mapping_candidates import propose_mapping_candidates
from .mapping_performance_compare import summarize_trials


def choose_exploration_cohort(proposals: dict, candidate_count: int = 6) -> list[dict]:
    """Pick a measured-baseline plus diverse families/directions from GGUF.

    Prefer a full GPU->host family (visible memory change), a single
    host->GPU weight (safe first promotion), then a same-family swap.
    This is a diagnostic cohort, not a performance rank.
    """
    if candidate_count < 1:
        raise ValueError("candidate_count deve essere positivo.")
    items = proposals["candidates"]
    baseline = next((c for c in items if c["id"] == "baseline"), None)
    if baseline is None:
        raise ValueError("Baseline assente.")
    family_bytes = defaultdict(int)
    # Keep the first, already validated screening cohort restricted to
    # repeating-block tensors. Output/embedding are a distinct high-impact
    # calibration phase and require an explicit native placement probe.
    pilot_families = {"attention", "feed_forward", "normalization", "other_block"}
    eligible_families = set()
    for c in items:
        if c["family"] not in pilot_families or c["kind"] != "single_tensor":
            continue
        if c["direction"] == "gpu_to_host":
            family_bytes[c["family"]] += c["gguf_bytes_to_host"]
            eligible_families.add(c["family"])
        elif c["direction"] == "host_to_gpu":
            # If no repeating blocks are offloaded, promotions must
            # still be discoverable even though there are no evictions.
            eligible_families.add(c["family"])
    families = sorted(eligible_families, key=lambda f: (-family_bytes[f], f))
    chosen = [baseline]
    used = {baseline["id"]}
    for family in families:
        for direction in ("gpu_to_host", "host_to_gpu", "swap"):
            available = [
                c for c in items
                if c["family"] == family and c["direction"] == direction
                and c["kind"] not in ("tied_input_output", "unknown_baseline_tensor")
                and c["id"] not in used
            ]
            if not available:
                continue
            if direction == "gpu_to_host":
                # Measure a meaningful family-level eviction first.
                available.sort(key=lambda c: (
                    0 if c["kind"] == "block_family" else 1,
                    -c["gguf_bytes_to_host"],
                    c["id"],
                ))
            elif direction == "host_to_gpu":
                # A smaller single promotion avoids excessive first-test VRAM demand.
                available.sort(key=lambda c: (
                    0 if c["kind"] == "single_tensor" else 1,
                    c["gguf_bytes_to_gpu"],
                    c["id"],
                ))
            else:
                available.sort(key=lambda c: (
                    abs(c["estimated_vram_weight_delta_bytes"]),
                    c["gguf_bytes_to_host"] + c["gguf_bytes_to_gpu"],
                    c["id"],
                ))
            winner = available[0]
            chosen.append(winner)
            used.add(winner["id"])
            if len(chosen) - 1 >= candidate_count:
                return chosen
    return chosen


def measured_pareto(summaries: list[dict]) -> list[str]:
    """Non-dominated *observed* configurations; no weighted magic score.

    Prefer higher decode, higher prefill and lower sampled VRAM peak.
    Rounding and experimental variance mean the frontier is provisional.
    """
    def dominates(a, b):
        metrics = (
            a["median_decode_tps"] >= b["median_decode_tps"],
            a["median_prefill_tps"] >= b["median_prefill_tps"],
            a["max_sampled_vram_peak_mib"] <= b["max_sampled_vram_peak_mib"],
        )
        strict = (
            a["median_decode_tps"] > b["median_decode_tps"]
            or a["median_prefill_tps"] > b["median_prefill_tps"]
            or a["max_sampled_vram_peak_mib"] < b["max_sampled_vram_peak_mib"]
        )
        return all(metrics) and strict

    return [
        item["candidate_id"] for item in summaries
        if not any(dominates(other, item) for other in summaries if other is not item)
    ]


def screen(gpu_layers: int = 32, candidate_count: int = 6,
           runs: int = 2, tokens: int = 128, timeout: int = 180) -> dict:
    if candidate_count < 1 or not 1 <= runs <= 5 or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("candidate_count>=1, runs=1..5, tokens=64..256, timeout>=1.")

    gpu = GPUManager.calibration_status()
    if not gpu["ready"]:
        raise RuntimeError(
            f"GPU non pronta: {gpu['errors']}; processi={gpu['foreign_processes']}"
        )
    layout = ModelManager().inspect("gemma-2-9b-it", n_ctx=512)
    proposals = propose_mapping_candidates(layout, gpu_layers)
    selected = choose_exploration_cohort(proposals, candidate_count)

    result = {
        "phase": "granular_mapping_exploration",
        "model": "gemma-2-9b-it",
        "method": "llama-cpp-python",
        "backend": "cuda",
        "gpu": gpu["gpu"]["name"],
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "generation_test_tokens": tokens,
        "repetitions": runs,
        "candidate_count_generated": proposals["candidate_count"],
        "candidate_count_tested_if_complete": len(selected),
        "candidates": [{
            "id": c["id"],
            "family": c["family"],
            "kind": c["kind"],
            "direction": c["direction"],
            "tensor_names": c["tensor_names"],
            "rules": c["rules"],
            "estimated_vram_weight_delta_bytes": c["estimated_vram_weight_delta_bytes"],
        } for c in selected],
        "trials": [],
        "summaries": [],
        "observed_pareto_ids": [],
        "success": False,
        "profile_generated": False,
        "note": (
            "Primo campionamento ragionato, non esplorazione esaustiva. "
            "Il fronte di Pareto usa throughput e VRAM misurati, senza "
            "punteggi pesati: piccole differenze possono essere rumore. "
            "Ogni override richiede verifica dai log nativi. "
            "La VRAM NVML e' campionata e n_ctx=512 non e' operativo."
        ),
    }

    for cycle in range(runs):
        order = selected if cycle % 2 == 0 else list(reversed(selected))
        for candidate in order:
            status = GPUManager.calibration_status()
            if not status["ready"]:
                result["error"] = (
                    f"GPU mutata durante lo screening: {status['errors']}; "
                    f"processi={status['foreign_processes']}"
                )
                return result
            observed = benchmark(
                gpu_layers=gpu_layers,
                max_tokens=tokens,
                timeout=timeout,
                candidate_id=candidate["id"] if candidate["rules"] else None,
            )
            mem = observed.get("measurements") or {}
            trial = {
                "run": cycle + 1,
                "candidate_id": candidate["id"],
                "success": observed.get("success") is True,
                "decode_tps": mem.get("decode_tokens_per_second"),
                "prefill_tps": mem.get("prefill_tokens_per_second"),
                "vram_peak_mib": mem.get("vram_peak_sampled_mib"),
                "generated_tokens": mem.get("generated_tokens"),
                "mapping_verification": observed.get("mapping_verification"),
                "worker_returncode": observed.get("worker_returncode"),
                "failure_type": observed.get("failure_type"),
                "native_signal": observed.get("native_signal"),
                "error": observed.get("error"),
            }
            result["trials"].append(trial)
            if not trial["success"]:
                result["error"] = (
                    f"Arresto prudenziale al candidato {candidate['id']} "
                    f"(prova {cycle+1}): "
                    f"{trial['error'] or trial['failure_type'] or 'verifica fallita'}"
                )
                return result

    result["summaries"] = summarize_trials(result["trials"], selected, runs)
    result["observed_pareto_ids"] = measured_pareto(result["summaries"])
    result["success"] = True
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Screening diagnostico automatico di mapping GGUF misti"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        report = screen(
            gpu_layers=args.gpu_layers,
            candidate_count=args.candidate_count,
            runs=args.runs, tokens=args.tokens, timeout=args.timeout,
        )
    except Exception as exc:
        report = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
