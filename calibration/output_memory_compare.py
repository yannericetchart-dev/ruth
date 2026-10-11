"""Compare CUDA model-buffer accounting versus actual NVML for Gemma 2 output.

Runs baseline and coupled-input/output-on-host using the same n_ctx=512.
No model profile is written and output-on-host is not assumed beneficial.
"""

import argparse
import json
import statistics

from hardware import GPUManager

from .benchmark import benchmark


HOST_ID = "tied:cpu:token_embd.weight"


def summarize_memory_audit(trials: list[dict], runs: int) -> dict:
    groups = {}
    for id_ in ("baseline", HOST_ID):
        rows = [t for t in trials if t["candidate_id"] == id_]
        if len(rows) != runs or any(not row["success"] for row in rows):
            raise ValueError(f"Audit incompleto o non valido: {id_}")
        metrics = (
            "decode_tps", "prefill_tps", "nvml_loaded_mib",
            "nvml_peak_mib", "native_gpu_model_mib", "load_seconds",
        )
        if any(
            not isinstance(t.get(key), (int, float)) or t[key] < 0
            for t in rows for key in metrics
        ):
            raise ValueError(f"Misure incomplete: {id_}")
        if not all(t.get("native_buffer_record_count", 0) >= 1 for t in rows):
            raise ValueError(f"Log di memoria nativa mancanti: {id_}")
        groups[id_] = {
            "median_decode_tps": round(statistics.median(t["decode_tps"] for t in rows), 2),
            "median_prefill_tps": round(statistics.median(t["prefill_tps"] for t in rows), 2),
            "median_load_seconds": round(statistics.median(t["load_seconds"] for t in rows), 3),
            "median_nvml_after_load_mib": round(statistics.median(t["nvml_loaded_mib"] for t in rows), 1),
            "max_nvml_peak_mib": max(t["nvml_peak_mib"] for t in rows),
            "median_native_gpu_model_buffer_mib": round(
                statistics.median(t["native_gpu_model_mib"] for t in rows), 2
            ),
            "median_native_host_model_buffer_mib": round(
                statistics.median(t["native_host_model_mib"] for t in rows), 2
            ),
        }

    baseline, host = groups["baseline"], groups[HOST_ID]
    return {
        "baseline": baseline,
        "tied_output_on_host": host,
        "decode_percent_change": round(
            (host["median_decode_tps"] / baseline["median_decode_tps"] - 1) * 100, 2
        ) if baseline["median_decode_tps"] else None,
        "prefill_percent_change": round(
            (host["median_prefill_tps"] / baseline["median_prefill_tps"] - 1) * 100, 2
        ) if baseline["median_prefill_tps"] else None,
        "nvml_peak_change_mib": round(
            host["max_nvml_peak_mib"] - baseline["max_nvml_peak_mib"], 1
        ),
        "nvml_after_load_change_mib": round(
            host["median_nvml_after_load_mib"] - baseline["median_nvml_after_load_mib"], 1
        ),
        "native_gpu_model_buffer_change_mib": round(
            host["median_native_gpu_model_buffer_mib"]
            - baseline["median_native_gpu_model_buffer_mib"], 2
        ),
        "native_host_model_buffer_change_mib": round(
            host["median_native_host_model_buffer_mib"]
            - baseline["median_native_host_model_buffer_mib"], 2
        ),
        "interpretation": (
            "Una riduzione del modello CUDA riportata nei log non dimostra "
            "una riduzione equivalente della VRAM NVML. Valutare entrambi."
        ),
    }


def compare(runs: int = 2, gpu_layers: int = 32, tokens: int = 128,
            timeout: int = 180) -> dict:
    if not 1 <= runs <= 5 or not 64 <= tokens <= 256 or timeout < 1:
        raise ValueError("runs=1..5, tokens=64..256, timeout>=1.")
    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(f"GPU non disponibile: {status['errors']}; {status['foreign_processes']}")

    report = {
        "phase": "tied_output_native_memory_audit",
        "model": "gemma-2-9b-it",
        "gpu_layers": gpu_layers,
        "probe_n_ctx": 512,
        "selected_n_ctx": None,
        "runs": runs,
        "max_generated_tokens": tokens,
        "trials": [],
        "comparison": None,
        "success": False,
        "profile_generated": False,
        "note": (
            "Verifica l'apparente mancato risparmio VRAM: log nativi dei "
            "buffer model/KV/compute/output e NVML totale. "
            "Non prova che i buffer nativi rappresentino tutta la VRAM. "
            "Le differenze di throughput su due prove sono preliminari."
        ),
    }
    for cycle in range(runs):
        order = ("baseline", HOST_ID) if cycle % 2 == 0 else (HOST_ID, "baseline")
        for id_ in order:
            current = GPUManager.calibration_status()
            if not current["ready"]:
                report["error"] = (
                    f"Stato GPU non valido: {current['errors']}; "
                    f"{current['foreign_processes']}"
                )
                return report

            measurement = benchmark(
                gpu_layers=gpu_layers,
                max_tokens=tokens,
                timeout=timeout,
                candidate_id=HOST_ID if id_ == HOST_ID else None,
                native_memory_audit=True,
            )
            mem = measurement.get("measurements") or {}
            native = measurement.get("native_memory_breakdown") or {}
            trial = {
                "run": cycle + 1,
                "candidate_id": id_,
                "success": measurement.get("success") is True,
                "decode_tps": mem.get("decode_tokens_per_second"),
                "prefill_tps": mem.get("prefill_tokens_per_second"),
                "load_seconds": mem.get("load_seconds"),
                "nvml_before_mib": mem.get("vram_used_before_mib"),
                "nvml_loaded_mib": mem.get("vram_used_loaded_mib"),
                "nvml_peak_mib": mem.get("vram_peak_sampled_mib"),
                "nvml_peak_by_phase_mib": mem.get("vram_peak_by_phase_mib"),
                "native_gpu_model_mib": native.get("gpu_model_buffer_mib"),
                "native_host_model_mib": native.get("host_model_buffer_mib"),
                "native_buffer_record_count": native.get("record_count"),
                "native_buffers_by_category_domain": native.get(
                    "totals_mib_by_category_and_domain"
                ),
                "native_records": native.get("native_buffer_records"),
                "mapping_verification": measurement.get("mapping_verification"),
                "failure_type": measurement.get("failure_type"),
                "native_signal": measurement.get("native_signal"),
                "error": measurement.get("error"),
            }
            report["trials"].append(trial)
            if not trial["success"]:
                report["error"] = (
                    f"Audit {id_} #{cycle + 1} fallito: "
                    f"{trial['error'] or trial['failure_type']}"
                )
                return report

    report["comparison"] = summarize_memory_audit(report["trials"], runs)
    report["success"] = True
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Attribuzione memoria per output Gemma 2 su CUDA o host"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        result = compare(args.runs, args.gpu_layers, args.tokens, args.timeout)
    except Exception as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
