"""Phase 2: test CPU KV baseline against selective CPU weight offload.

All weights reside on CUDA during the *reference*, while host KV lets
inference run beyond the phase-1 all-GPU context. Each candidate moves
one homologous block-tensor group to CPU to attempt GPU-resident KV at
the EXACT same enlarged context, prompt length and 128 decode tokens.
Native placement and KV residency are mandatory; CPU vs GPU throughput
is measured, not inferred. No phase-1 reports or operational profiles
are changed.
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

from .gpu_context import verify_gpu_runtime
from .baseline_discovery import minimal_chat_prompt
from .mapping_grouping import (
    group_inventory, choose_group_trials, native_group_trial,
)
from .mapping_native_validation import verify_override_logs
from .native_memory import parse_native_memory
from .worker_failure import analyze_worker_failure


SCHEMA_VERSION = 4
WORKER_REPETITIONS = 3
PROBE_BATCH = 32
PROBE_GENERATED_TOKENS = 128
BASELINE_MODE = "baseline_host_kv"
PERFORMANCE_TOLERANCE = 0.03
MAX_MEDIAN_RELATIVE_DEVIATION = 0.05


def benchmark_context(reference_ctx: int,
                      model_context_limit: int | None = None) -> tuple[int, int]:
    """First strictly larger power-of-two context, bounded by GGUF limit."""
    if (not isinstance(reference_ctx, int) or isinstance(reference_ctx, bool)
        or reference_ctx < 64):
        raise ValueError("Mapping: riferimento contesto non valido.")
    if (model_context_limit is not None
        and (not isinstance(model_context_limit, int)
             or model_context_limit < 1)):
        raise ValueError("Mapping: limite contesto GGUF non valido.")
    next_power = 1 << reference_ctx.bit_length()
    upper = model_context_limit if model_context_limit else next_power
    chosen = min(next_power, upper)
    if chosen <= reference_ctx or chosen <= PROBE_GENERATED_TOKENS + 16:
        raise ValueError(
            "Mapping: impossibile provare un contesto superiore al massimo "
            "full-GPU entro il limite architetturale."
        )
    return chosen, PROBE_GENERATED_TOKENS


def make_mapping_tokens(llm, architecture: str, n_ctx: int,
                        generated_tokens: int, model_id: str) -> list[int]:
    """Same prompt length across host-KV baseline and mapped GPU-KV trials."""
    if (generated_tokens < 1 or generated_tokens > 128
        or n_ctx - generated_tokens < 17):
        raise ValueError("Mapping: prompt e generazione non entrano in n_ctx.")
    target = n_ctx - generated_tokens - 1
    base = minimal_chat_prompt(architecture, model_id)
    reps = max(1, n_ctx // 4)
    while reps <= n_ctx * 64:
        prompt = base + (
            " Analizza la sequenza ed esponi il ragionamento con precisione."
            * reps
        )
        ids = llm.tokenize(
            prompt.encode("utf-8"), add_bos=True, special=True
        )
        if len(ids) >= target:
            return list(ids[:target])
        reps *= 2
    raise RuntimeError("Mapping: impossibile costruire prompt richiesto.")


def verify_mapping_kv_runtime(native: dict, *, host_kv: bool) -> dict:
    """Verify host-KV baseline or GPU-KV candidate from actual native logs."""
    if not host_kv:
        return verify_gpu_runtime(native)
    records = native.get("native_buffer_records") or []
    cache = [r for r in records if r.get("category") == "kv"]
    compute = [r for r in records if r.get("category") == "compute"]
    host = [r for r in cache if r.get("domain") in (
        "host_cpu", "host_cuda_pinned"
    )]
    other = [r for r in cache if r.get("domain") not in (
        "host_cpu", "host_cuda_pinned"
    )]
    valid = bool(cache) and len(host) == len(cache) and bool(compute)
    return {
        "verified": valid,
        "kv": {
            "verified_host_resident": bool(host) and not other,
            "host_buffer_records": len(host),
            "other_buffer_records": len(other),
        },
        "compute": {
            "verified_present": bool(compute),
            "gpu_buffer_records": sum(
                r.get("domain") == "gpu_vram" for r in compute
            ),
        },
        "note": "KV nativa RAM/CPU verificata; GPU model weights checked separately.",
    }


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
    if group_id != BASELINE_MODE and (
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
        "offload_kqv": group_id != BASELINE_MODE,
        "kv_target": "host" if group_id == BASELINE_MODE else "gpu",
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
                offload_kqv=(group_id != BASELINE_MODE),
            )
            report["measurements"]["load_s"] = round(
                time.perf_counter() - t0, 5
            )
            report["measurements"]["vram_after_load_mib"] = used_mib()
            tokens = make_mapping_tokens(
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
                         *, n_ctx: int, tokens: int,
                         timeout: int = 300) -> dict:
    """Execute exactly one baseline or group trial in a fresh CUDA process."""
    group_id = trial["id"]
    if group_id != BASELINE_MODE and not group_id.startswith("group:"):
        raise ValueError("Mapping: ID gruppo non valido.")
    command = [
        sys.executable, "-m", "calibration.mapping_phase",
        "--worker", "--model-path", str(model_path),
        "--model", model_id, "--mode", group_id,
        "--ctx", str(n_ctx), "--tokens", str(tokens),
    ]
    try:
        child = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False, "mode": group_id, "failure_class": "timeout",
            "error": f"Mapping: worker scaduto dopo {timeout}s",
        }
    try:
        payload = json.loads(child.stdout)
    except (TypeError, ValueError):
        payload = None
    frame_ok = isinstance(payload, dict)
    report = payload if frame_ok else {}
    report["mode"] = group_id
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
    worker_ok = (
        child.returncode == 0 and frame_ok
        and report.get("worker_ok") is True
    )
    report.update(analyze_worker_failure(
        child.stderr, report.get("error"),
        child.returncode, worker_ok,
    ))
    evidence = verify_override_logs(trial, child.stderr)
    report["native_placement"] = evidence
    memory = parse_native_memory(child.stderr)
    report["native_memory"] = memory
    host_kv = group_id == BASELINE_MODE
    runtime = verify_mapping_kv_runtime(memory, host_kv=host_kv)
    report["gpu_runtime_validation"] = runtime
    measured = report.get("measurements") or {}
    metrics = ("prefill_tokens_per_s", "decode_tokens_per_s",
               "vram_peak_sampled_mib", "host_peak_rss_mib")
    finite = all(
        isinstance(measured.get(k), (int, float))
        and not isinstance(measured[k], bool)
        and math.isfinite(measured[k])
        and measured[k] > 0
        for k in metrics
    )
    success = (
        worker_ok and evidence["verified"] and runtime["verified"]
        and report.get("n_ctx") == n_ctx
        and report.get("requested_tokens") == tokens
        and report.get("model_id") == model_id
        and report.get("mode") == group_id
        and report.get("offload_kqv") is (not host_kv)
        and report.get("kv_target") == ("host" if host_kv else "gpu")
        and measured.get("prompt_tokens") == n_ctx - tokens - 1
        and measured.get("generated_tokens") == tokens
        and finite and not measured.get("nvml_errors")
    )
    report["success"] = success
    if success:
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
    if not success and not report.get("error"):
        report["error"] = (
            "Mapping: placement, KV, RAM/VRAM oppure throughput non verificati."
        )
    return report


def _median_stats(trials: list[dict]) -> dict:
    metrics = (
        "vram_peak_sampled_mib", "host_peak_rss_mib",
        "prefill_tokens_per_s", "decode_tokens_per_s",
    )
    values = {}
    for key in metrics:
        sample = [
            t["measurements"][key] for t in trials
            if t.get("success") is True
        ]
        if len(sample) != WORKER_REPETITIONS:
            raise ValueError("Mapping: mancano campioni validati per il benchmark.")
        median = statistics.median(sample)
        mad = statistics.median(abs(x - median) for x in sample)
        values[key] = {
            "median": round(median, 4),
            "relative_median_absolute_deviation": round(mad / median, 5),
            "samples": sample,
        }
    return values


def _compare_with_baseline(reference: dict, current: dict) -> dict:
    """Same enlarged context: host KV full weights vs GPU KV + CPU tensors.

    A higher candidate VRAM peak is not a failure: GPU KV deliberately
    occupies VRAM that the host-KV baseline leaves free. Acceptance is
    governed by throughput and stability, not an irrelevant negative
    'VRAM saved relative to host-KV' requirement.
    """
    baseline_vram = reference["vram_peak_sampled_mib"]["median"]
    trial_vram = current["vram_peak_sampled_mib"]["median"]
    vram_delta = round(trial_vram - baseline_vram, 3)
    deltas = {}
    stable = True
    for label in ("prefill_tokens_per_s", "decode_tokens_per_s"):
        original = reference[label]["median"]
        test = current[label]["median"]
        deltas[label] = round(100 * (test / original - 1), 3)
        stable = stable and all(
            stats[label]["relative_median_absolute_deviation"]
            <= MAX_MEDIAN_RELATIVE_DEVIATION
            for stats in (reference, current)
        )
    performance_ok = all(
        percent >= -100 * PERFORMANCE_TOLERANCE
        for percent in deltas.values()
    )
    speed_benefit = any(
        percent >= 100 * PERFORMANCE_TOLERANCE
        for percent in deltas.values()
    )
    if not stable:
        verdict = "inconclusive_variability"
    elif not performance_ok:
        verdict = "rejected_performance_regression"
    elif speed_benefit:
        verdict = "promising_extended_context_gpu_kv_speedup"
    else:
        verdict = "no_measurable_speed_gain"
    return {
        "vram_delta_vs_host_kv_baseline_mib": vram_delta,
        "prefill_change_percent": deltas["prefill_tokens_per_s"],
        "decode_change_percent": deltas["decode_tokens_per_s"],
        "host_peak_rss_change_mib": round(
            current["host_peak_rss_mib"]["median"]
            - reference["host_peak_rss_mib"]["median"], 3
        ),
        "no_material_throughput_regression": performance_ok if stable else None,
        "stable_enough_for_preliminary_comparison": stable,
        "verdict": verdict,
        "performance_tolerance_percent": PERFORMANCE_TOLERANCE * 100,
        "note": (
            "Baseline full-weight CUDA/host KV confrontata alla stessa "
            "lunghezza e prompt con CPU weight group/GPU KV. Il delta "
            "VRAM non equivale alla VRAM liberata dai pesi: cambia anche "
            "la residenza KV. Se manca un guadagno di velocita' misurabile "
            "il gruppo non viene promosso. Nessun profilo operativo."
        ),
    }



def _reusable_trials(
    samples: object, group_id: str, *, model_id: str,
    n_ctx: int, tokens: int,
) -> bool:
    """Reuse only complete, positive native and throughput evidence."""
    if not isinstance(samples, list) or len(samples) != WORKER_REPETITIONS:
        return False
    for row in samples:
        if not isinstance(row, dict):
            return False
        measured = row.get("measurements") or {}
        if (
            row.get("success") is not True
            or row.get("mode") != group_id
            or row.get("model_id") != model_id
            or row.get("n_ctx") != n_ctx
            or row.get("requested_tokens") != tokens
            or row.get("offload_kqv") is (group_id != BASELINE_MODE)
            or row.get("kv_target") != (
                "host" if group_id == BASELINE_MODE else "gpu"
            )
            or row.get("worker_returncode") != 0
            or (row.get("native_placement") or {}).get("verified") is not True
            or (row.get("gpu_runtime_validation") or {}).get("verified") is not True
            or (row.get("gpu_runtime_validation") or {}).get(
                "kv", {}
            ).get(
                "verified_host_resident"
                if group_id == BASELINE_MODE else "verified_gpu_resident"
            ) is not True
            or measured.get("prompt_tokens") != n_ctx - tokens - 1
            or measured.get("generated_tokens") != tokens
            or measured.get("nvml_errors")
        ):
            return False
        for key in (
            "vram_peak_sampled_mib", "host_peak_rss_mib",
            "prefill_tokens_per_s", "decode_tokens_per_s",
        ):
            value = measured.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                return False
    return True


def _compatible_previous_report(previous: object, current: dict) -> bool:
    """Only schema4 with the same extended context and KV mode is reusable."""
    if not isinstance(previous, dict):
        return False
    if previous.get("schema_version") != SCHEMA_VERSION:
        return False
    if previous.get("status") not in (
        "diagnostic_group_comparison_completed", "benchmark_in_progress",
    ):
        return False
    for key in (
        "model_id", "source", "repository", "quantization", "filename",
        "model_path", "shard_filenames", "block_count", "tensor_count",
        "probe_n_ctx", "probe_tokens", "n_batch",
        "reference_full_cuda_n_ctx", "workload",
        "native_model_context_limit", "baseline_kv_target",
        "candidate_kv_target", "baseline_offload_kqv",
        "candidate_offload_kqv", "repeat_processes_per_variant",
    ):
        if previous.get(key) != current.get(key):
            return False
    return True


def run_mapping_phase(baseline: dict, matrix: dict, context: dict,
                      *, model_id: str, progress=None,
                      timeout: int = 300,
                      previous_report: dict | None = None,
                      checkpoint=None) -> dict:
    """Repeated baseline, then every eligible homologous block tensor group."""
    if (
        not isinstance(baseline, dict)
        or baseline.get("model_id") != model_id
        or baseline.get("native_verified_all_gpu") is not True
        or not isinstance(matrix, dict)
        or matrix.get("success") is not True
        or matrix.get("status") != "enumeration_only"
        or not isinstance(context, dict)
        or context.get("success") is not True
        or not isinstance(context.get("confirmed_max_gpu_resident_n_ctx"), int)
    ):
        raise RuntimeError(
            "Mapping: fase 1 full-CUDA incompleta per questo modello."
        )
    path = Path(baseline["local_path"]).resolve()
    if (
        not path.is_file()
        or path.name != baseline["filename"]
        or matrix.get("filename") != baseline["filename"]
        or Path(matrix.get("model_path", "/invalid")).resolve() != path
        or matrix.get("block_count", 0) < 1
        or context.get("filename") != baseline["filename"]
        or context.get("quantization") != baseline.get("quantization")
    ):
        raise RuntimeError(
            "Mapping: baseline, matrice e contesto non coerenti."
        )
    shard_names = [p.name for p in gguf_file_set(path)]
    if (baseline.get("shard_filenames")
        and shard_names != baseline["shard_filenames"]):
        raise RuntimeError("Mapping: shard GGUF diversi dalla fase 1.")
    arch = baseline.get("architecture")
    header = inspect_gguf(path, n_ctx=64)
    if header["metadata"].get("general.architecture") != arch:
        raise RuntimeError("Mapping: architettura GGUF diversa dalla baseline.")
    limit = header["metadata"].get(f"{arch}.context_length")
    if not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("Mapping: contesto massimo GGUF sconosciuto.")
    n_ctx, tokens = benchmark_context(
        context["confirmed_max_gpu_resident_n_ctx"], limit
    )
    layout = inspect_gguf(path, n_ctx=n_ctx)
    if (
        layout["metadata"].get("general.architecture") != arch
        or layout["metadata"].get(f"{arch}.block_count")
            != matrix["block_count"]
        or layout.get("weights_bytes")
            != matrix.get("serialized_weight_bytes")
    ):
        raise RuntimeError("Mapping: struttura GGUF modificata dalla fase 1.")

    groups = group_inventory(layout)
    selected = choose_group_trials(groups)
    control = native_group_trial(layout, None)
    control["id"] = BASELINE_MODE
    control["kv_target"] = "host"
    plan = [control]
    for group in selected:
        candidate = native_group_trial(layout, group)
        candidate["kv_target"] = "gpu"
        plan.append(candidate)
    report = {
        "schema_version": SCHEMA_VERSION, "phase": "mapping",
        "status": "benchmark_in_progress",
        "success": False, "model_id": model_id,
        "source": baseline.get("source"),
        "repository": baseline.get("repository"),
        "quantization": baseline.get("quantization"),
        "filename": path.name, "model_path": str(path),
        "shard_filenames": shard_names,
        "block_count": matrix["block_count"],
        "tensor_count": len(layout["tensors"]),
        "probe_n_ctx": n_ctx, "probe_tokens": tokens,
        "n_batch": PROBE_BATCH,
        "reference_full_cuda_n_ctx": context[
            "confirmed_max_gpu_resident_n_ctx"
        ],
        "workload": "extended_context_host_kv_baseline_vs_gpu_kv_group_128_decode",
        "native_model_context_limit": limit,
        "baseline_kv_target": "host",
        "candidate_kv_target": "gpu",
        "baseline_offload_kqv": False,
        "candidate_offload_kqv": True,
        "repeat_processes_per_variant": WORKER_REPETITIONS,
        "screening_size_threshold_mib": None,
        "group_trial_limit": None,
        "groups": groups,
        "screened_group_count": len(groups),
        "eligible_group_count": sum(
            g["eligible_for_trial"] for g in groups
        ),
        "tested_group_ids": [g["id"] for g in selected],
        "baseline_trials": [],
        "baseline_statistics": None,
        "group_evaluations": [],
        "promising_group_ids": [],
        "selected_mapping": None,
        "profile_generated": False,
        "runtime_optimized": False, "context_tuned": False,
        "note": (
            "Byte GGUF unici e shape FLOP proxy ordinano gruppi omologhi "
            "ma non dimostrano risparmio VRAM o velocita'. Baseline: "
            "pesi full-CUDA e KV host; gruppi: tensori selezionati su CPU "
            "e KV su CUDA, tutti allo STESSO n_ctx maggiore del massimo "
            "fase1 e con 128 token decode, 3 processi ciascuno. "
            "Nessuna configurazione operativa viene scelta."
        ),
    }
    from copy import deepcopy

    previous = (
        previous_report
        if _compatible_previous_report(previous_report, report)
        else None
    )
    prior_groups = {
        item["group_id"]: item
        for item in (previous or {}).get("group_evaluations", [])
        if isinstance(item, dict) and isinstance(item.get("group_id"), str)
    }

    def persist_progress():
        if checkpoint is not None:
            checkpoint(report)

    for position, trial in enumerate(plan):
        group_id = trial["id"]
        existing = (
            (previous or {}).get("baseline_trials")
            if position == 0 else
            (prior_groups.get(group_id) or {}).get("trials")
        )
        reused = _reusable_trials(
            existing, group_id, model_id=model_id,
            n_ctx=n_ctx, tokens=tokens,
        )
        if reused:
            samples = deepcopy(existing)
            report["reused_native_trial_count"] = (
                report.get("reused_native_trial_count", 0) + len(samples)
            )
            if progress is not None:
                progress({
                    "event": "mapping_trial_reused",
                    "mode": group_id,
                    "native_trials": len(samples),
                })
        else:
            samples = []
            for repetition in range(1, WORKER_REPETITIONS + 1):
                if progress is not None:
                    progress({
                        "event": "mapping_probe_start", "mode": group_id,
                        "repetition": repetition,
                        "repetitions": WORKER_REPETITIONS,
                        "host_tensors": trial["host_tensor_count"],
                        "gpu_tensors": trial["gpu_tensor_count"],
                    })
                outcome = probe_native_mapping(
                    path, model_id, trial, n_ctx=n_ctx, tokens=tokens,
                    timeout=timeout,
                )
                samples.append(outcome)
                if position == 0:
                    report["baseline_trials"] = samples
                if progress is not None:
                    progress({
                        "event": "mapping_probe_result", "mode": group_id,
                        "repetition": repetition,
                        "success": outcome.get("success"),
                        "failure_class": outcome.get("failure_class"),
                        "vram_peak_mib": (
                            outcome.get("measurements") or {}
                        ).get("vram_peak_sampled_mib"),
                    })
                if not outcome.get("success"):
                    # Full-CUDA reference must succeed. A failed individual
                    # group is measured evidence of unsuitability, not a
                    # reason to skip every subsequent eligible group.
                    if position == 0:
                        report["status"] = "blocked_baseline_native_benchmark"
                        report["error"] = (
                            f"Mapping baseline ripetizione {repetition} "
                            f"non validata: {outcome.get('failure_class')}; "
                            f"{outcome.get('error')}"
                        )
                        persist_progress()
                        return report
                    report["group_evaluations"].append({
                        "group_id": group_id,
                        "trials": samples,
                        "statistics": None,
                        "comparison": None,
                        "status": "native_probe_failed",
                        "failure_class": outcome.get("failure_class"),
                        "error": outcome.get("error"),
                    })
                    if progress is not None:
                        progress({
                            "event": "mapping_group_failed",
                            "mode": group_id,
                            "failure_class": outcome.get("failure_class"),
                        })
                    persist_progress()
                    break
            if len(samples) < WORKER_REPETITIONS or not all(
                row.get("success") is True for row in samples
            ):
                # A native failure in a group has been recorded above;
                # continue to explore the rest of the tensor families.
                continue

        stats = _median_stats(samples)
        if position == 0:
            report["baseline_trials"] = samples
            report["baseline_statistics"] = stats
            persist_progress()
            if progress is not None:
                progress({
                    "event": "mapping_baseline_complete",
                    "prefill_tok_s": stats["prefill_tokens_per_s"]["median"],
                    "decode_tok_s": stats["decode_tokens_per_s"]["median"],
                    "vram_peak_mib": stats["vram_peak_sampled_mib"]["median"],
                })
            continue

        comparison = _compare_with_baseline(
            report["baseline_statistics"], stats
        )
        report["group_evaluations"].append({
            "group_id": group_id,
            "trials": samples, "statistics": stats,
            "comparison": comparison,
            "status": comparison["verdict"],
        })
        if comparison["verdict"] == "promising_extended_context_gpu_kv_speedup":
            report["promising_group_ids"].append(group_id)
        if progress is not None:
            progress({
                "event": "mapping_group_comparison",
                "mode": group_id,
                "verdict": comparison["verdict"],
                "vram_delta_mib": comparison["vram_delta_vs_host_kv_baseline_mib"],
                "prefill_percent": comparison["prefill_change_percent"],
                "decode_percent": comparison["decode_change_percent"],
            })
        persist_progress()

    report["success"] = True
    report["status"] = "diagnostic_group_comparison_completed"
    report["evaluated_group_count"] = len(report["group_evaluations"])
    report["native_failed_group_count"] = sum(
        entry["status"] == "native_probe_failed"
        for entry in report["group_evaluations"]
    )
    persist_progress()
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Worker isolato di benchmark mapping Ruth (fase 2)"
    )
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-path")
    parser.add_argument("--model")
    parser.add_argument("--mode")
    parser.add_argument("--ctx", type=int)
    parser.add_argument("--tokens", type=int)
    args = parser.parse_args(argv)
    if (not args.worker or not args.model_path or not args.model
        or not args.mode or not args.ctx or not args.tokens):
        parser.error("Modulo riservato al worker nativo della fase mapping.")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        report = _worker(
            Path(args.model_path), args.model, args.mode,
            args.ctx, args.tokens,
        )
    except Exception as exc:
        report = {
            "worker_ok": False, "mode": args.mode,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report.get("worker_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
