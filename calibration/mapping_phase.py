"""Phase 2: measured homologous GGUF tensor-group offload search.

Baseline: all weights on CUDA, measured repeatedly with identical prompt,
context and batch. Screen all block.N.<tensor-role> families using unique
GGUF bytes and a deliberately approximate compute proxy; test at most three
eligible groups, each in three independent native CUDA workers.

Only group tensor sets move to host. Every other weight, including tied
Gemma embedding/output, remains CUDA; native logs must verify all occurrences.
The goal is usable VRAM without a reproducible performance regression.
This phase does NOT tune context, select a production profile or overwrite
phase-1 baseline/context reports.
"""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import threading
import time

from hardware import GPUManager
from model.inspect import inspect_gguf, gguf_file_set

from .baseline_discovery import minimal_chat_prompt
from .gpu_context import verify_gpu_runtime, make_context_tokens
from .mapping_grouping import (
    group_inventory, choose_group_trials, native_group_trial,
    MAX_TRIAL_GROUPS, MIN_GROUP_MIB,
)
from .mapping_native_validation import verify_override_logs
from .native_memory import parse_native_memory
from .worker_failure import analyze_worker_failure


SCHEMA_VERSION = 2
WORKER_REPETITIONS = 3
MIN_N_CTX = 128
MAX_N_CTX = 512
PROBE_BATCH = 32
PROBE_TOKENS_CAP = 32
PERFORMANCE_TOLERANCE = 0.03
MAX_MEDIAN_RELATIVE_DEVIATION = 0.05


def benchmark_context(reference_ctx: int) -> tuple[int, int]:
    """Fixed fair workload for the same model/quantization and all trials."""
    if not isinstance(reference_ctx, int) or reference_ctx < MIN_N_CTX:
        raise ValueError("Mapping: riferimento n_ctx full-CUDA insufficiente.")
    n_ctx = min(MAX_N_CTX, max(MIN_N_CTX, reference_ctx // 2))
    tokens = min(PROBE_TOKENS_CAP, max(12, n_ctx // 8))
    return n_ctx, tokens


def _worker(model_path: Path, model_id: str, group_id: str,
            n_ctx: int, token_count: int) -> dict:
    """Fixed workload in isolated worker, native stderr retained externally."""
    import pynvml
    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

    if not llama_supports_gpu_offload():
        raise RuntimeError("Mapping: backend CUDA llama.cpp non disponibile.")
    status = GPUManager.calibration_status()
    if not status.get("ready"):
        raise RuntimeError("Mapping: GPU non pronta.")
    gpu = status["gpu"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError("Mapping: CUDA_VISIBLE_DEVICES diversa dalla GPU.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    arch = layout["metadata"]["general.architecture"]
    groups = group_inventory(layout)
    selected = next((g for g in groups if g["id"] == group_id), None)
    if group_id != "baseline" and (
        selected is None or not selected["eligible_for_trial"]
    ):
        raise ValueError("Mapping: gruppo non valido o non ammesso alla prova.")
    trial = native_group_trial(layout, selected)
    block_count = layout["metadata"][f"{arch}.block_count"]
    report = {
        "worker_ok": False, "mode": group_id, "model_id": model_id,
        "n_ctx": n_ctx, "n_batch": PROBE_BATCH,
        "requested_tokens": token_count,
        "n_gpu_layers": block_count + 1,
        "host_tensor_count": trial["host_tensor_count"],
        "gpu_tensor_count": trial["gpu_tensor_count"],
        "measurements": {}, "error": None,
    }
    stop = threading.Event()
    samples = []
    errors = []
    llm = None
    pynvml.nvmlInit()
    try:
        device = pynvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])

        def used_mib():
            return round(
                pynvml.nvmlDeviceGetMemoryInfo(device).used / 1048576, 2
            )

        report["measurements"]["vram_before_mib"] = used_mib()

        def monitor():
            while not stop.is_set():
                try:
                    samples.append(used_mib())
                except pynvml.NVMLError as exc:
                    errors.append(type(exc).__name__)
                    break
                stop.wait(0.05)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            t0 = time.perf_counter()
            llm, _ = LlamaCppPythonMappingAdapter().load(
                model_path, n_gpu_layers=block_count + 1,
                rules=trial["rules"], n_ctx=n_ctx, n_batch=PROBE_BATCH,
                n_ubatch=PROBE_BATCH,
            )
            report["measurements"]["load_s"] = round(
                time.perf_counter() - t0, 5
            )
            report["measurements"]["vram_after_load_mib"] = used_mib()
            tokens = make_context_tokens(
                llm, arch, n_ctx, token_count, model_id=model_id
            )
            report["measurements"]["prompt_tokens"] = len(tokens)
            llm.reset()
            t0 = time.perf_counter()
            llm.eval(tokens)
            prefill_s = time.perf_counter() - t0
            output = llm.generate([], reset=False, temp=0.0, top_k=1)
            t0 = time.perf_counter()
            generated = 0
            try:
                for _ in range(token_count):
                    next(output)
                    generated += 1
            finally:
                output.close()
            decode_s = time.perf_counter() - t0
            metrics = report["measurements"]
            metrics["prefill_s"] = round(prefill_s, 5)
            metrics["decode_s"] = round(decode_s, 5)
            metrics["generated_tokens"] = generated
            metrics["prefill_tokens_per_s"] = round(
                len(tokens) / prefill_s, 3
            ) if prefill_s > 0 else None
            metrics["decode_tokens_per_s"] = round(
                generated / decode_s, 3
            ) if decode_s > 0 else None
            report["worker_ok"] = generated == token_count
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if llm is not None:
                llm.close()
            stop.set()
            thread.join(timeout=3)
            report["measurements"]["vram_peak_sampled_mib"] = max(
                [report["measurements"]["vram_before_mib"]] + samples
            )
            report["measurements"]["host_peak_rss_mib"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
            )
            report["measurements"]["nvml_errors"] = errors
            if errors:
                report["worker_ok"] = False
                report["error"] = "Mapping: campionamento NVML incompleto."
    finally:
        pynvml.nvmlShutdown()
    return report


def probe_native_mapping(model_path: Path, model_id: str, trial: dict,
                         timeout: int = 300) -> dict:
    """One fixed-runtime placement per short-lived CUDA subprocess."""
    if trial.get("id") not in MODES:
        raise ValueError("Mapping: candidato non registrato.")
    cmd = [
        sys.executable, "-m", "calibration.mapping_phase",
        "--worker", "--model-path", str(model_path),
        "--model", model_id, "--mode", trial["id"],
    ]
    try:
        child = subprocess.run(
            cmd, text=True, capture_output=True, timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False, "mode": trial["id"], "failure_class": "timeout",
            "error": f"Mapping: worker scaduto dopo {timeout}s",
        }
    try:
        parsed = json.loads(child.stdout)
    except (TypeError, ValueError):
        parsed = None
    frame_ok = isinstance(parsed, dict)
    report = parsed if frame_ok else {}
    report["mode"] = trial["id"]
    report["worker_json_valid"] = frame_ok
    report["worker_returncode"] = child.returncode
    if child.returncode < 0:
        try:
            report["worker_signal"] = signal.Signals(
                -child.returncode
            ).name
        except ValueError:
            report["worker_signal"] = "UNKNOWN_SIGNAL"
    else:
        report["worker_signal"] = None

    reported_success = (
        child.returncode == 0 and frame_ok
        and report.get("worker_ok") is True
    )
    report.update(analyze_worker_failure(
        child.stderr, report.get("error"),
        child.returncode, reported_success,
    ))
    evidence = verify_override_logs(trial, child.stderr)
    report["native_placement"] = evidence
    native_memory = parse_native_memory(child.stderr)
    report["native_memory"] = native_memory
    runtime = verify_gpu_runtime(native_memory)
    report["gpu_runtime_validation"] = runtime
    measured = report.get("measurements") or {}
    valid = (
        reported_success
        and report.get("n_ctx") == PROBE_N_CTX
        and report.get("model_id") == model_id
        and evidence["verified"]
        and runtime["verified"]
        and measured.get("generated_tokens") == PROBE_TOKENS
        and isinstance(measured.get("vram_peak_sampled_mib"), (int, float))
        and math.isfinite(measured["vram_peak_sampled_mib"])
        and measured["vram_peak_sampled_mib"] > 0
        and not measured.get("nvml_errors")
    )
    report["success"] = valid
    if valid:
        report["failure_class"] = None
    elif report.get("suspected_cuda_oom"):
        report["failure_class"] = "native_cuda_oom"
    elif report.get("native_cuda_resource_allocation_failed"):
        report["failure_class"] = "native_cuda_resource_allocation_failure"
    elif not frame_ok:
        report["failure_class"] = "worker_protocol_error"
    elif not evidence["verified"]:
        report["failure_class"] = "tensor_placement_not_verified"
    elif not runtime["verified"]:
        report["failure_class"] = "gpu_kv_compute_not_verified"
    else:
        report["failure_class"] = "native_mapping_runtime_failure"
    if not valid and not report.get("error"):
        report["error"] = (
            "Mapping: placement, KV CUDA o inferenza non verificati."
        )
    return report


def run_mapping_phase(baseline: dict, matrix: dict, context: dict,
                      *, model_id: str, progress=None,
                      timeout: int = 300) -> dict:
    """Bounded phase-2 pilot only; never select or persist an operational map."""
    if (not isinstance(baseline, dict)
        or baseline.get("model_id") != model_id
        or baseline.get("native_verified_all_gpu") is not True
        or not isinstance(matrix, dict)
        or matrix.get("success") is not True
        or matrix.get("status") != "enumeration_only"
        or not isinstance(context, dict)
        or context.get("success") is not True
        or not isinstance(context.get("confirmed_max_gpu_resident_n_ctx"), int)):
        raise RuntimeError(
            "Mapping: fase 1 non completamente verificata per questo modello."
        )
    path = Path(baseline["local_path"])
    if (not path.is_file()
        or path.name != baseline["filename"]
        or matrix.get("filename") != baseline["filename"]
        or Path(matrix["model_path"]) != path.resolve()
        or matrix.get("block_count", 0) < 1
        or context.get("filename") != baseline["filename"]
        or context.get("quantization") != baseline.get("quantization")):
        raise RuntimeError("Mapping: baseline, matrice e contesto non coerenti.")
    layout = inspect_gguf(path, n_ctx=PROBE_N_CTX)
    if (layout["metadata"].get("general.architecture") != baseline["architecture"]
        or layout["metadata"].get(
            baseline["architecture"] + ".block_count"
        ) != matrix["block_count"]
        or layout.get("weights_bytes") != matrix.get("serialized_weight_bytes")):
        raise RuntimeError("Mapping: struttura GGUF diversa dalla fase 1.")
    trials = native_mapping_trials(layout)
    report = {
        "schema_version": SCHEMA_VERSION, "phase": "mapping",
        "status": "diagnostic_native_placement_pending",
        "model_id": model_id, "source": baseline.get("source"),
        "repository": baseline.get("repository"),
        "quantization": baseline.get("quantization"),
        "filename": path.name, "model_path": str(path.resolve()),
        "block_count": matrix["block_count"],
        "tensor_count": len(layout["tensors"]),
        "probe_n_ctx": PROBE_N_CTX, "probe_tokens": PROBE_TOKENS,
        "reference_full_cuda_n_ctx": context["confirmed_max_gpu_resident_n_ctx"],
        "trials": [], "selected_mapping": None,
        "runtime_optimized": False, "context_tuned": False,
        "profile_generated": False, "success": False,
        "note": (
            "Due prove diagnostiche reali: prefisso CPU di un blocco "
            "e singolo tensore attention su CPU, tutti gli altri pesi "
            "sulla GPU, KV e compute CUDA. I tempi non sono benchmark. "
            "Nessun mapping operativo selezionato."
        ),
    }
    for trial in trials:
        if progress is not None:
            progress({
                "event": "mapping_probe_start", "mode": trial["id"],
                "host_tensors": trial["host_tensor_count"],
                "gpu_tensors": trial["gpu_tensor_count"],
            })
        outcome = probe_native_mapping(
            path, model_id, trial, timeout=timeout,
        )
        report["trials"].append(outcome)
        if progress is not None:
            progress({
                "event": "mapping_probe_result", "mode": trial["id"],
                "success": outcome.get("success"),
                "failure_class": outcome.get("failure_class"),
                "vram_peak_mib": (outcome.get("measurements") or {}).get(
                    "vram_peak_sampled_mib"
                ),
            })
        if not outcome.get("success"):
            report["status"] = "blocked_unverified_native_mapping"
            report["error"] = (
                f"Mapping {trial['id']} non validato: "
                f"{outcome.get('failure_class')}; {outcome.get('error')}"
            )
            return report
    report["success"] = True
    report["status"] = "diagnostic_native_placement_validated"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Worker diagnostico mapping CPU/GPU, non profilo operativo"
    )
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args(argv)
    if not args.worker or not args.model_path or not args.model or not args.mode:
        parser.error("Modulo riservato al worker di mapping fase 2.")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        report = _worker(Path(args.model_path), args.model, args.mode)
    except Exception as exc:
        report = {
            "worker_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("worker_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
