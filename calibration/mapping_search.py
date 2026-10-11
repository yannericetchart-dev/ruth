"""Discover CUDA-feasible block offloads without choosing an operational context.

This is a bounded feasibility search, NOT a performance optimizer.
Each trial runs in a fresh process, so a failed GPU allocation cannot
poison the Python process running the search.
"""

import argparse
import json
import subprocess
import sys
import time

from hardware import GPUManager
from model import ModelManager


def _run_trial(layers: int, timeout_seconds: int) -> dict:
    command = [
        sys.executable, "-m", "calibration.model_probe",
        "--gpu-layers", str(layers),
    ]
    started = time.perf_counter()
    try:
        child = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "gpu_layers": layers, "status": "timeout",
            "seconds": round(time.perf_counter() - started, 2),
            "error": f"Timeout dopo {timeout_seconds} secondi",
        }

    report = None
    try:
        report = json.loads(child.stdout)
    except (json.JSONDecodeError, TypeError):
        pass

    result = {
        "gpu_layers": layers,
        "seconds": round(time.perf_counter() - started, 2),
        "status": "unknown_failure",
        "returncode": child.returncode,
    }
    if child.returncode < 0:
        import signal

        result["status"] = "native_crash"
        try:
            result["native_signal"] = signal.Signals(-child.returncode).name
        except ValueError:
            result["native_signal"] = str(-child.returncode)

    if report is not None:
        measurement = report.get("measurement") or {}
        details = measurement.get("measurements") or {}
        result["vram_peak_mib"] = details.get("gpu_used_peak_sampled_mib")
        result["vram_before_mib"] = details.get("gpu_used_before_mib")
        result["vram_after_close_mib"] = details.get("gpu_used_after_close_mib")
        result["reported_tps_including_prefill"] = details.get(
            "tokens_per_second_including_prefill"
        )
        if child.returncode == 0 and report.get("candidate_valid") is True:
            result["status"] = "feasible"
            return result

    # An unsuccessful model load can mean insufficient memory, but it can
    # also mean a driver, file or backend problem. Do not silently conflate.
    error = (report or {}).get("measurement", {}).get("error") or ""
    combined = (error + "\n" + child.stderr)[-3000:]
    result["error"] = (error or child.stderr.strip() or "Risposta del probe non valida")[-600:]

    resource_markers = (
        "out of memory", "cuda error out of memory", "cudamalloc failed",
        "cuda error 2", "failed to allocate", "failed to create llama_context",
    )
    if report is not None and any(marker in combined.lower() for marker in resource_markers):
        # A generic context-creation failure is only a *possible* memory
        # limit. Mark it explicitly instead of claiming proven CUDA OOM.
        result["status"] = "probable_resource_limit"
    return result


def search(timeout_seconds: int = 180) -> dict:
    if timeout_seconds < 1:
        raise ValueError("Il timeout deve essere positivo.")

    before = GPUManager.calibration_status()
    if not before["ready"]:
        raise RuntimeError(
            f"GPU non pronta: {before['errors']}; "
            f"processi esterni={before['foreign_processes']}"
        )

    model_id = "gemma-2-9b-it"
    layout = ModelManager().inspect(model_id, n_ctx=512)
    arch = layout["metadata"].get("general.architecture")
    count = layout["metadata"].get(f"{arch}.block_count")
    if not isinstance(count, int) or count <= 0:
        raise RuntimeError("Numero di blocchi non determinabile dal GGUF.")

    report = {
        "phase": "model_mapping_feasibility",
        "model": model_id,
        "engine": "llama.cpp",
        "method": "llama-cpp-python",
        "backend": "cuda",
        "mapping_capability": "contiguous_gpu_layer_offload",
        "block_count": count,
        "gpu": before["gpu"]["name"],
        "vram_total_mib": before["memory"]["total_mib"],
        "probe_context_note": (
            "I singoli trial usano n_ctx=512 come requisito tecnico. "
            "Il contesto operativo NON viene selezionato."
        ),
        "trials": [],
        "max_verified_gpu_layers": None,
        "selected_n_ctx": None,
        "profile_generated": False,
        "search_complete": False,
        "stop_reason": None,
        "notes": (
            "Ricerca binaria della fattibilita' dei blocchi GPU, non ottimizzazione "
            "della velocita' o qualita'. Un errore di creazione del contesto "
            "non dimostra da solo un OOM."
        ),
    }

    lo, hi = 1, count
    # For standard llama.cpp block offload, GPU weight allocations should
    # increase with n_gpu_layers. An unexpected error aborts this assumption.
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = _run_trial(mid, timeout_seconds)
        report["trials"].append(trial)

        if trial["status"] == "feasible":
            report["max_verified_gpu_layers"] = mid
            lo = mid + 1
        elif trial["status"] == "probable_resource_limit":
            hi = mid - 1
        else:
            report["stop_reason"] = (
                f"Ricerca interrotta al candidato {mid}: {trial['status']}. "
                "Non classificato come limite VRAM."
            )
            break

        # An unexpected foreign GPU process invalidates the comparisons.
        current = GPUManager.calibration_status()
        if not current["ready"]:
            report["stop_reason"] = (
                f"Ambiente GPU modificato: {current['errors']}; "
                f"processi esterni={current['foreign_processes']}"
            )
            break

    if report["stop_reason"] is None:
        report["search_complete"] = True
        if report["max_verified_gpu_layers"] is None:
            report["stop_reason"] = "Nessun mapping provato e' risultato fattibile."
        else:
            report["stop_reason"] = "Intervallo di fattibilita' esplorato."

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ricerca automatica preliminare dei layer GPU fattibili"
    )
    parser.add_argument("--timeout", type=int, default=180, help="Secondi massimi per trial")
    args = parser.parse_args()
    try:
        result = search(args.timeout)
    except Exception as exc:
        print(f"Ricerca non avviata: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["search_complete"] or result["max_verified_gpu_layers"] is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
