"""Automatic comparison of CUDA model placements already proven feasible.

A diagnostic survey, not a validated Ruth profile. n_ctx=512 is solely
the small technical context of the model benchmark; context sizing comes later.
"""

import argparse
import json
import statistics
import sys

from hardware import GPUManager

from .benchmark import benchmark
from .mapping_search import search


def compare(runs: int = 2, tokens: int = 128, timeout: int = 180) -> dict:
    if not 1 <= runs <= 5 or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("runs: 1-5; tokens: 64-256; timeout: positivo.")

    baseline = GPUManager.calibration_status()
    if not baseline["ready"]:
        raise RuntimeError(
            f"GPU non pronta: {baseline['errors']}; "
            f"processi esterni={baseline['foreign_processes']}"
        )

    # Obtain candidates dynamically from model structure and GPU feasibility.
    # A CUDA crash in a higher candidate does not make the lower successful
    # trials disappear, but means the feasible upper bound is not proven.
    feasibility = search(timeout_seconds=timeout)
    candidates = sorted({
        trial["gpu_layers"]
        for trial in feasibility["trials"] if trial["status"] == "feasible"
    })
    if not candidates:
        raise RuntimeError(
            f"La ricerca di fattibilita' non ha trovato candidati: "
            f"{feasibility['stop_reason']}"
        )

    report = {
        "phase": "model_mapping_performance_survey",
        "model": feasibility["model"],
        "engine": feasibility["engine"],
        "method": feasibility["method"],
        "backend": feasibility["backend"],
        "gpu": feasibility["gpu"],
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "generation_test_tokens": tokens,
        "repetitions_per_candidate": runs,
        "feasibility_search_complete": feasibility["search_complete"],
        "feasibility_stop_reason": feasibility["stop_reason"],
        "feasibility_trials": feasibility["trials"],
        "compared_layers": candidates,
        "comparisons": [],
        "fastest_measured_gpu_layers": None,
        "survey_complete": False,
        "profile_generated": False,
        "stop_reason": None,
        "notes": (
            "Classifica temporanea solo per velocita' di decode. "
            "Non verifica qualita', n_ctx operativo, stabilita' lunga, "
            "mapping granulare dei tensori o VRAM minima di profilo. "
            "Il campionamento NVML non cattura necessariamente ogni picco."
        ),
    }

    for layers in candidates:
        entry = {
            "gpu_layers": layers,
            "runs": [],
            "status": "incomplete",
            "median_decode_tokens_per_second": None,
            "median_prefill_tokens_per_second": None,
            "peak_vram_mib_max": None,
            "minimum_sampled_free_vram_mib": None,
        }
        report["comparisons"].append(entry)

        for index in range(runs):
            state = GPUManager.calibration_status()
            if not state["ready"]:
                report["stop_reason"] = (
                    f"Ambiente GPU modificato: {state['errors']}; "
                    f"processi esterni={state['foreign_processes']}"
                )
                break

            result = benchmark(gpu_layers=layers, max_tokens=tokens, timeout=timeout)
            measurements = result.get("measurements") or {}
            entry["runs"].append({
                "run": index + 1,
                "success": result.get("success", False),
                "decode_tokens_per_second": measurements.get("decode_tokens_per_second"),
                "prefill_tokens_per_second": measurements.get("prefill_tokens_per_second"),
                "generated_tokens": measurements.get("generated_tokens"),
                "vram_peak_sampled_mib": measurements.get("vram_peak_sampled_mib"),
                "worker_returncode": result.get("worker_returncode"),
                "failure_type": result.get("failure_type"),
                "native_signal": result.get("native_signal"),
                "suspected_cuda_oom": result.get("suspected_cuda_oom"),
                "cuda_diagnostics": result.get("cuda_diagnostics"),
                "error": result.get("error"),
            })
            if not result.get("success"):
                entry["status"] = "failed"
                break

        if report["stop_reason"] is not None:
            break

        readings = entry["runs"]
        if len(readings) != runs or any(not run["success"] for run in readings):
            entry["status"] = "failed"
            continue

        decode = [run["decode_tokens_per_second"] for run in readings]
        prefill = [run["prefill_tokens_per_second"] for run in readings]
        peaks = [run["vram_peak_sampled_mib"] for run in readings]
        if any(x is None or x <= 0 for x in decode + prefill + peaks):
            entry["status"] = "invalid_measurement"
            continue

        entry["status"] = "benchmarked"
        entry["median_decode_tokens_per_second"] = round(statistics.median(decode), 2)
        entry["median_prefill_tokens_per_second"] = round(statistics.median(prefill), 2)
        entry["peak_vram_mib_max"] = max(peaks)
        entry["minimum_sampled_free_vram_mib"] = round(
            baseline["memory"]["total_mib"] - max(peaks), 1
        )

    confirmed = [x for x in report["comparisons"] if x["status"] == "benchmarked"]
    if confirmed:
        fastest = max(
            confirmed,
            key=lambda x: (x["median_decode_tokens_per_second"], x["gpu_layers"]),
        )
        report["fastest_measured_gpu_layers"] = fastest["gpu_layers"]

    if report["stop_reason"] is None:
        report["survey_complete"] = True
        report["stop_reason"] = "Confronto concluso dei candidati rilevati."

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confronta automaticamente mapping CUDA senza scegliere n_ctx"
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    try:
        result = compare(runs=args.runs, tokens=args.tokens, timeout=args.timeout)
    except Exception as exc:
        print(f"Confronto non avviato: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["survey_complete"] or result["fastest_measured_gpu_layers"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
