"""Ruth stage 2: rank singleton offloads, load-only verify at 512, then infer.

A 256-token NVML delta is NOT an upper bound on saving at 512. Therefore
there is NO UNSOUND hard-pruning of candidates based on a memory proxy.
Every individually profiled GGUF block tensor gets a 512 load-only attempt
in descending order of *measured* 256-token memory relief.

The load worker constructs and closes llama.cpp; never generates a token.
On positive native-GPU-KV load verification, regular mapping inference
with identical weights/KV is benchmarked separately at 512.
Checkpoint is atomic and recoverable at EVERY individual substage.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time

from hardware import GPUManager
from model.inspect import inspect_gguf

from .gpu_context import context_limit, verify_native_kv_config
from .mapping_grouping import native_tensor_trial
from .mapping_native_validation import verify_override_logs
from .mapping_phase import PROBE_BATCH, probe_native_mapping, verify_mapping_kv_runtime
from .native_memory import parse_native_memory
from .worker_failure import analyze_worker_failure
from .kv_adaptive_tensor_offload import _probe_summary, summarize_probe_repeats
from .kv_single_tensor_census import _native_verified, _tensor_inventory
from .profile_pipeline import atomic_json


SCHEMA = "ruth-phase2-single-to-512-v1"
CENSUS_SCHEMA = "ruth-phase2-single-tensor-census-v1"


def _load_worker(model_path: Path, model_id: str, names: list[str],
                 n_ctx: int) -> dict:
    """Isolated CUDA worker: instantiate llama context only. No eval/decode."""
    import pynvml
    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

    if not llama_supports_gpu_offload():
        raise RuntimeError("Backend CUDA llama.cpp non disponibile.")
    state = GPUManager.calibration_status()
    if not state.get("ready"):
        raise RuntimeError("GPU non pronta: " + repr(state.get("errors")))
    gpu = state["gpu"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError("CUDA_VISIBLE_DEVICES non corrisponde a GPU NVML.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    if n_ctx < 64 or n_ctx > context_limit(layout):
        raise ValueError("n_ctx fuori limite GGUF.")
    arch = layout["metadata"]["general.architecture"]
    n_gpu_layers = layout["metadata"][f"{arch}.block_count"] + 1
    trial = native_tensor_trial(layout, names)
    output = {
        "worker_ok": False, "model_id": model_id,
        "mode": trial["id"], "host_tensor_names": sorted(names),
        "n_ctx": n_ctx, "n_gpu_layers": n_gpu_layers,
        "n_batch": PROBE_BATCH, "n_ubatch": PROBE_BATCH,
        "offload_kqv": True, "kv_target": "gpu",
        "flash_attn": False, "kv_type_k": "f16", "kv_type_v": "f16",
        "evaluated_tokens": 0, "generated_tokens": 0,
        "measurements": {}, "error": None,
    }
    samples, nvml_errors = [], []
    stop = threading.Event()
    llm = None
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        measurements = output["measurements"]
        measurements["vram_total_mib"] = round(info.total / 1048576, 2)

        def used():
            return round(
                pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1048576, 2
            )

        measurements["vram_before_mib"] = used()

        def monitor():
            while not stop.is_set():
                try:
                    samples.append(used())
                except pynvml.NVMLError as exc:
                    nvml_errors.append(type(exc).__name__)
                    break
                stop.wait(.02)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            start = time.perf_counter()
            llm, _ = LlamaCppPythonMappingAdapter().load(
                model_path, n_gpu_layers=n_gpu_layers, rules=trial["rules"],
                n_ctx=n_ctx, n_batch=PROBE_BATCH, n_ubatch=PROBE_BATCH,
                offload_kqv=True, flash_attn=False,
                type_k="f16", type_v="f16",
            )
            measurements["load_s"] = round(time.perf_counter() - start, 5)
            measurements["vram_after_load_mib"] = used()
            # Logical and CUDA/native validation happen in the parent from
            # the *real* stderr, not from claims in this JSON payload.
            output["worker_ok"] = True
        except Exception as exc:
            output["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            stop.set()
            thread.join(timeout=3)
            peak = max(
                [measurements["vram_before_mib"]]
                + samples
                + ([measurements["vram_after_load_mib"]]
                   if "vram_after_load_mib" in measurements else [])
            )
            measurements["vram_peak_load_only_mib"] = peak
            measurements["vram_load_peak_increment_mib"] = round(
                peak - measurements["vram_before_mib"], 2
            )
            measurements["vram_headroom_at_load_peak_mib"] = round(
                measurements["vram_total_mib"] - peak, 2
            )
            measurements["nvml_errors"] = nvml_errors
            if nvml_errors:
                output["worker_ok"] = False
                output["error"] = "NVML non verificabile durante load."
            if llm is not None:
                llm.close()
            measurements["host_peak_rss_mib"] = round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2
            )
    finally:
        pynvml.nvmlShutdown()
    return output


def probe_load_only(model_path: Path, model_id: str, trial: dict,
                    *, n_ctx: int, timeout: int = 600) -> dict:
    """Run a fresh CUDA subprocess, verify mapping and KV without inference."""
    cmd = [
        sys.executable, "-m", "calibration.kv_single_to_512",
        "--worker", "--model-path", str(model_path),
        "--model-id", model_id, "--n-ctx", str(n_ctx),
        "--tensor-names", json.dumps(trial["host_tensor_names"]),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {
            "success": False, "failure_class": "timeout",
            "error": f"Load-only scaduto dopo {timeout} secondi",
            "n_ctx": n_ctx,
            "host_tensor_names": trial["host_tensor_names"],
            "mapping_id": trial["id"],
        }
    try:
        payload = json.loads(proc.stdout)
        if not isinstance(payload, dict):
            raise ValueError("Worker non ha restituito un oggetto.")
    except (TypeError, ValueError):
        payload = {"worker_ok": False, "error": "Worker JSON non valido."}
    placement = verify_override_logs(trial, proc.stderr)
    native = parse_native_memory(proc.stderr)
    gpu_kv = verify_mapping_kv_runtime(native, host_kv=False)
    precision = verify_native_kv_config(
        proc.stderr, flash_attn=False, type_k="f16", type_v="f16"
    )
    measurement = payload.get("measurements") or {}
    load_ok = bool(
        proc.returncode == 0
        and payload.get("worker_ok") is True
        and payload.get("mode") == trial["id"]
        and payload.get("host_tensor_names") == trial["host_tensor_names"]
        and payload.get("n_ctx") == n_ctx
        and payload.get("evaluated_tokens") == 0
        and payload.get("generated_tokens") == 0
        and placement["verified"] and gpu_kv["verified"]
        and precision["verified"]
        and not measurement.get("nvml_errors")
        and all(
            isinstance(measurement.get(key), (int, float))
            and math.isfinite(measurement[key])
            for key in ("vram_before_mib", "vram_after_load_mib",
                        "vram_peak_load_only_mib", "vram_total_mib",
                        "vram_load_peak_increment_mib")
        )
        and measurement.get("vram_peak_load_only_mib", 0)
            <= measurement.get("vram_total_mib", 0)
        and measurement.get("vram_load_peak_increment_mib", -1) >= 0
    )
    failure = analyze_worker_failure(
        proc.stderr, payload.get("error"), proc.returncode, load_ok
    )
    return {
        "success": load_ok,
        "n_ctx": n_ctx, "host_tensor_names": trial["host_tensor_names"],
        "mapping_id": trial["id"],
        "worker_returncode": proc.returncode,
        "worker_signal": (-proc.returncode if proc.returncode < 0 else None),
        "failure_class": (
            None if load_ok else
            "native_cuda_oom" if failure["suspected_cuda_oom"] else
            "native_cuda_resource_allocation_failure"
            if failure["native_cuda_resource_allocation_failed"]
            else failure.get("failure_hint") or "load_or_native_verification_failed"
        ),
        "error": payload.get("error"),
        "native_placement_verified": placement["verified"],
        "gpu_kv_verified": gpu_kv["verified"],
        "native_kv_type_verified": precision["verified"],
        "native_memory_totals_mib":
            native.get("totals_mib_by_category_and_domain"),
        "native_kv_buffer_records": [
            row for row in native.get("native_buffer_records", [])
            if row["category"] == "kv"
        ],
        "measurements": measurement,
        "failure_diagnostics": failure,
        "interpretation": (
            "Success proves a particular 512 CUDA context can be "
            "constructed with GPU KV, not that inference succeeds. "
            "Observed load-only peak and increment describe this RUN, "
            "not the mathematical minimum required VRAM across runs."
        ),
    }


def _median_native_load_mib(row: dict, *, label: str) -> float:
    """True post-load NVML usage, not inference peak or GGUF tensor size.

    Each independent source sample must have an actual load measurement
    and valid native GPU-KV evidence; aggregates do not expose load-only
    VRAM and must not be inferred from inference peaks.
    """
    import statistics

    if row.get("success") is not True:
        raise ValueError(f"{label}: profilo nativo non riuscito.")
    samples = row.get("samples")
    if (not isinstance(samples, list) or len(samples) < 2
        or row.get("repetitions") != len(samples)):
        raise ValueError(f"{label}: campioni di caricamento insufficienti.")
    values = []
    for sample in samples:
        if (sample.get("success") is not True
            or sample.get("native_placement_verified") is not True
            or sample.get("gpu_kv_verified") is not True
            or sample.get("native_kv_type_verified") is not True
            or sample.get("nvml_errors")):
            raise ValueError(f"{label}: placement o KV GPU non verificata.")
        load = sample.get("vram_after_load_mib")
        if (not isinstance(load, (int, float))
            or isinstance(load, bool) or
            not math.isfinite(load) or load <= 0):
            raise ValueError(f"{label}: vram_after_load_mib non verificabile.")
        values.append(load)
    return float(statistics.median(values))


def order_single_candidates(census: dict, *, target_ctx: int,
                            allow_partial: bool = False) -> list[dict]:
    """Rank by observed post-load GPU relief at 256; NO hard pruning."""
    params = census.get("parameters") or {}
    if census.get("schema") != CENSUS_SCHEMA:
        raise ValueError("Schema census incompatibile.")
    ctx = params.get("n_ctx")
    if type(ctx) is not int or target_ctx <= ctx:
        raise ValueError("Target deve superare n_ctx del census.")
    result = census.get("tensor_results") or {}
    count = census.get("eligible_tensor_count")
    if (type(count) is not int or count < 1 or
        len(result) != count or
        (not allow_partial and census.get("completed") is not True)):
        raise ValueError(
            "Censimento non completo: nessun tensore puo essere "
            "silenziosamente scartato. Terminare prima il census."
        )
    baseline = census.get("baseline_control") or {}
    if not isinstance(baseline.get("metrics"), dict):
        raise ValueError("Baseline n_ctx di origine non verificata.")
    baseline_load = _median_native_load_mib(baseline, label="baseline")
    baseline_peak = baseline["metrics"]["vram_peak_mib"]
    reused = census.get("reused_from_greedy") or {}
    new = census.get("new_measurements") or {}
    ranked = []
    for name, data in result.items():
        if not data.get("success") or not isinstance(data.get("metrics"), dict):
            if not allow_partial:
                raise ValueError("Misura mancante/non valida per: " + name)
            continue
        source = reused if data.get("source") == "greedy_v3" else new
        raw = source.get(name)
        if not isinstance(raw, dict):
            if not allow_partial:
                raise ValueError("Campioni nativi mancanti per: " + name)
            continue
        try:
            load = _median_native_load_mib(raw, label=name)
        except ValueError:
            if not allow_partial:
                raise
            continue
        peak = data["metrics"]["vram_peak_mib"]
        gain_load = round(baseline_load - load, 3)
        gain_peak = round(baseline_peak - peak, 3)
        ranked.append({
            "tensor": name,
            "source_n_ctx": ctx,
            "measured_load_256_mib": load,
            "baseline_load_256_mib": baseline_load,
            "measured_load_gain_256_mib": gain_load,
            "measured_peak_256_mib": peak,
            "measured_peak_gain_256_mib": gain_peak,
            "measured_prefill_256_tok_s":
                data["metrics"]["prefill_tok_s"],
            "measured_decode_256_tok_s":
                data["metrics"]["decode_tok_s"],
            "hard_excluded": False,
            "reason": (
                "La misura NVML dopo load a 256 non e un limite "
                "superiore dimostrato del risparmio al target."
            ),
        })
    ranked.sort(key=lambda row: (
        -row["measured_load_gain_256_mib"],
        -row["measured_peak_gain_256_mib"], row["tensor"]
    ))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked

def _valid_loaded_configuration(row: dict, *,
                                n_ctx: int, trial: dict) -> bool:
    """Require native proof and numerical NVML evidence to promote a load."""
    if (not isinstance(row, dict)
        or row.get("success") is not True
        or row.get("n_ctx") != n_ctx
        or row.get("host_tensor_names") != trial["host_tensor_names"]
        or row.get("mapping_id") != trial["id"]
        or row.get("native_placement_verified") is not True
        or row.get("gpu_kv_verified") is not True
        or row.get("native_kv_type_verified") is not True):
        return False
    m = row.get("measurements") or {}
    keys = ("vram_before_mib", "vram_after_load_mib",
            "vram_peak_load_only_mib", "vram_total_mib",
            "vram_load_peak_increment_mib")
    return (all(
        isinstance(m.get(k), (int, float))
        and not isinstance(m[k], bool)
        and math.isfinite(m[k]) and m[k] >= 0
        for k in keys
    ) and m["vram_after_load_mib"] <= m["vram_peak_load_only_mib"]
      and m["vram_before_mib"] <= m["vram_peak_load_only_mib"]
      and m["vram_peak_load_only_mib"] <= m["vram_total_mib"]
      and not m.get("nvml_errors"))


def run_single_to_512(
    *, census_path: Path, output: Path, target_ctx: int = 512,
    max_new_loads: int | None = None, max_new_benchmarks: int | None = None,
    benchmark_repeats: int = 2, timeout: int = 600,
    retry_failed_inferences: int = 0, retry_only: bool = False,
    progress=None, load_probe=probe_load_only,
    inference_probe=probe_native_mapping,
) -> dict:
    """Plan once, load-only once each, infer only on native-validated loads."""
    if (type(benchmark_repeats) is not int
        or not 1 <= benchmark_repeats <= 10
        or type(retry_failed_inferences) is not int
        or retry_failed_inferences < 0
        or type(retry_only) is not bool
        or (retry_only and retry_failed_inferences == 0)
        or any(v is not None and (type(v) is not int or v < 1)
            for v in (max_new_loads, max_new_benchmarks))):
        raise ValueError("Budget o ripetizioni non validi.")
    source = Path(census_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output == source:
        raise ValueError("Non sovrascrivere il censimento.")
    raw = source.read_bytes()
    census = json.loads(raw)
    ranked = order_single_candidates(census, target_ctx=target_ctx)
    p = census["parameters"]
    if (p.get("flash_attn") is not False
        or p.get("type_k") != "f16"
        or p.get("type_v") != "f16"
        or p.get("n_ctx") >= target_ctx):
        raise ValueError("Census deve usare KV FP16 GPU e FA OFF.")
    model_path = Path(p["model_path"]).resolve()
    model_id = p["model_id"]
    layout = inspect_gguf(model_path, n_ctx=target_ctx)
    if target_ctx > context_limit(layout):
        raise ValueError("n_ctx superiore al limite GGUF.")
    names = sorted(row["tensor"] for row in ranked)
    if names != _tensor_inventory(layout):
        raise ValueError("Inventario GGUF diverso dal census.")
    identity = {
        "census_sha256": hashlib.sha256(raw).hexdigest(),
        "model_id": model_id, "model_path": str(model_path),
        "source_n_ctx": p["n_ctx"], "target_n_ctx": target_ctx,
        "generated_tokens": p["generated_tokens"],
        "flash_attn": False, "type_k": "f16", "type_v": "f16",
        "benchmark_repeats": benchmark_repeats, "timeout": timeout,
        "candidate_count": len(ranked),
        "selection_rule": "all_singles_descending_verified_post_load_256_gain_peak_tiebreak_v2",
    }
    initial = {}
    if output.exists():
        previous = json.loads(output.read_text("utf-8"))
        if previous.get("schema") != SCHEMA or previous.get("parameters") != identity:
            raise ValueError("Checkpoint incompatibile: usare un output distinto.")
        initial = previous
    entries = initial.get("results") or {}
    if not isinstance(entries, dict) or any(name not in names for name in entries):
        raise ValueError("Checkpoint risultati malformato.")
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            raise ValueError("Checkpoint malformato: " + name)
        trial = native_tensor_trial(layout, [name])
        load = entry.get("load_only")
        infer = entry.get("inference")
        if load is not None and (
            not isinstance(load, dict)
            or load.get("mapping_id") != trial["id"]
            or load.get("n_ctx") != target_ctx
            or load.get("host_tensor_names") != [name]
            or (load.get("success") is True
                and not _valid_loaded_configuration(
                    load, n_ctx=target_ctx, trial=trial))
        ):
            raise ValueError("Checkpoint load non valido: " + name)
        if infer is not None and (
            load is None or load.get("success") is not True
            or not isinstance(infer, dict)
            or infer.get("host_tensor_names") != [name]
            or infer.get("n_ctx") != target_ctx
            or infer.get("mapping_id") != trial["id"]
            or (infer.get("success") is True
                and not _native_verified(
                    infer, n_ctx=target_ctx, names=[name],
                    trial_id=trial["id"],
                    repeats=benchmark_repeats,
                    tokens=p["generated_tokens"]))
        ):
            raise ValueError("Checkpoint benchmark non valido: " + name)
    # Optional diagnostic replay: keep the original failure intact in
    # inference_history and only replay previously FAILED inferences.
    # Runtime controls intentionally do not alter the checkpoint identity.
    # The user can inspect old versus new CUDA stderr in one report.
    retried_names = set()
    for item in ranked:
        if len(retried_names) >= retry_failed_inferences:
            break
        name = item["tensor"]
        entry = entries.get(name)
        if not entry:
            continue
        prior = entry.get("inference")
        if (not isinstance(prior, dict) or prior.get("success") is not False
            or entry.get("load_only", {}).get("success") is not True):
            continue
        history = entry.setdefault("inference_history", [])
        if not isinstance(history, list):
            raise ValueError("Checkpoint history non valido: " + name)
        history.append(entry.pop("inference"))
        retried_names.add(name)
    control = initial.get("control_load_only")
    report = {
        "schema": SCHEMA, "parameters": identity,
        "status": "in_progress", "completed": False,
        "candidate_order": ranked, "control_load_only": control,
        "results": entries, "summary": {},
        "hard_pruning": {
            "certified_exclusions": [],
            "count": 0,
            "explanation": (
                "Delta NVML misurato a 256 NON e un upper bound dimostrato "
                "del saving a 512. Nessuna esclusione preventiva certa. "
                "Caricamenti falliti sono esclusi dai benchmark, non dalla "
                "futura ricerca per coppie/terne."
            ),
        },
    }

    def save():
        loads_done = sum("load_only" in e for e in entries.values())
        completed_benchmark = sum(
            e.get("inference") is not None
            for e in entries.values()
        )
        successful_loads = [
            (name, e["load_only"])
            for name, e in entries.items()
            if isinstance(e.get("load_only"), dict)
            and e["load_only"].get("success") is True
        ]
        good_loads = len(successful_loads)
        min_peak = min(
            successful_loads,
            key=lambda pair: pair[1]["measurements"][
                "vram_peak_load_only_mib"
            ],
            default=None,
        )
        min_increment = min(
            successful_loads,
            key=lambda pair: pair[1]["measurements"][
                "vram_load_peak_increment_mib"
            ],
            default=None,
        )
        good_bench = sum(
            isinstance(e.get("inference"), dict)
            and e["inference"].get("success") is True
            for e in entries.values()
        )
        report["summary"] = {
            "candidates": len(ranked), "load_attempted": loads_done,
            "load_success": good_loads,
            "load_failures": loads_done-good_loads,
            "inference_attempted": completed_benchmark,
            "inference_success": good_bench,
            "remaining_loads": len(ranked)-loads_done,
            "remaining_inference": good_loads-completed_benchmark,
            "ready_for_phase3": (
                loads_done == len(ranked)
                and completed_benchmark == good_loads
            ),
            "has_successful_single_tensor": good_bench > 0,
            "minimum_observed_successful_load_peak_mib": (
                min_peak[1]["measurements"]["vram_peak_load_only_mib"]
                if min_peak else None
            ),
            "minimum_observed_successful_load_peak_tensor": (
                min_peak[0] if min_peak else None
            ),
            "minimum_observed_successful_load_increment_mib": (
                min_increment[1]["measurements"][
                    "vram_load_peak_increment_mib"
                ] if min_increment else None
            ),
            "minimum_observed_successful_load_increment_tensor": (
                min_increment[0] if min_increment else None
            ),
            "minimum_observed_load_note": (
                "Minimo osservato tra i caricamenti RIUSCITI di questa "
                "esecuzione, non una dimostrazione della VRAM "
                "minima teorica per creare il contesto."
            ),
        }
        done = report["summary"]["ready_for_phase3"] and control is not None
        report["completed"] = done
        report["status"] = (
            "completed_singleton_benchmarks" if done and good_bench
            else "completed_no_singleton_inference" if done
            else "partial_checkpoint"
        )
        atomic_json(output, report)

    save()
    # One load-only all-GPU control at target: valuable even if it fails.
    # It is NOT a benchmark and never promotes an invalid GPU configuration.
    if control is None:
        base_trial = native_tensor_trial(layout, [])
        try:
            control = load_probe(
                model_path, model_id, base_trial, n_ctx=target_ctx,
                timeout=timeout,
            )
        except Exception as exc:
            control = {
                "success": False, "n_ctx": target_ctx,
                "host_tensor_names": [], "mapping_id": base_trial["id"],
                "failure_class": "load_probe_exception",
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["control_load_only"] = control
        save()
        if progress:
            progress({
                "event": "control_load_only",
                "n_ctx": target_ctx, "success": control.get("success"),
                "failure_class": control.get("failure_class"),
            })

    new_loads = 0
    new_benchmarks = 0
    for item in ranked:
        name = item["tensor"]
        if retry_only and name not in retried_names:
            continue
        entry = entries.setdefault(name, {})
        trial = native_tensor_trial(layout, [name])
        if "load_only" not in entry:
            if retry_only:
                raise ValueError("Retry diagnostico senza caricamento verificato.")
            if max_new_loads is not None and new_loads >= max_new_loads:
                break
            try:
                entry["load_only"] = load_probe(
                    model_path, model_id, trial, n_ctx=target_ctx,
                    timeout=timeout,
                )
            except Exception as exc:
                entry["load_only"] = {
                    "success": False, "n_ctx": target_ctx,
                    "host_tensor_names": [name], "mapping_id": trial["id"],
                    "failure_class": "load_probe_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            new_loads += 1
            save()
            if progress:
                progress({
                    "event": "load_only", "rank": item["rank"],
                    "tensor": name,
                    "success": entry["load_only"].get("success"),
                    "observed_peak_mib": (
                        entry["load_only"].get("measurements") or {}
                    ).get("vram_peak_load_only_mib"),
                    "remaining_loads": report["summary"]["remaining_loads"],
                })
        if entry["load_only"].get("success") is not True:
            continue
        if not _valid_loaded_configuration(
            entry["load_only"], n_ctx=target_ctx, trial=trial
        ):
            raise ValueError(
                "Caricamento con prove GPU-KV native insufficienti: " + name
            )
        if "inference" in entry:
            continue
        if (max_new_benchmarks is not None
            and new_benchmarks >= max_new_benchmarks):
            continue
        observations = []
        for _ in range(benchmark_repeats):
            try:
                raw_result = inference_probe(
                    model_path, model_id, trial,
                    n_ctx=target_ctx, tokens=p["generated_tokens"],
                    timeout=timeout, flash_attn=False,
                    type_k="f16", type_v="f16",
                )
                observations.append(_probe_summary(raw_result))
            except Exception as exc:
                observations.append({
                    "success": False, "failure_class": "inference_probe_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if observations[-1]["success"] is not True:
                break
        aggregate = summarize_probe_repeats(observations, benchmark_repeats)
        aggregate.update({
            "n_ctx": target_ctx, "host_tensor_names": [name],
            "mapping_id": trial["id"],
        })
        if aggregate["success"] and not _native_verified(
            aggregate, n_ctx=target_ctx, names=[name],
            trial_id=trial["id"], repeats=benchmark_repeats,
            tokens=p["generated_tokens"],
        ):
            aggregate["success"] = False
            aggregate["failure_class"] = "unverified_native_cuda_inference"
            aggregate.pop("metrics", None)
        if aggregate["success"]:
            metrics = aggregate["metrics"]
            metrics["combined_time_s"] = round(
                (target_ctx-p["generated_tokens"]) /
                metrics["prefill_tok_s"] +
                p["generated_tokens"]/metrics["decode_tok_s"], 6
            )
        entry["inference"] = aggregate
        new_benchmarks += 1
        save()
        if progress:
            progress({
                "event": "inference", "rank": item["rank"],
                "tensor": name, "success": aggregate["success"],
                "decode_tok_s": aggregate.get("metrics", {}).get("decode_tok_s"),
                "remaining_inference":
                    report["summary"]["remaining_inference"],
            })
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fase 2: singoli tensori a 512, load-only poi inferenza."
    )
    parser.add_argument("--worker", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--n-ctx", type=int)
    parser.add_argument("--tensor-names")
    parser.add_argument("--census", type=Path,
                        default=Path("phase2_single_tensor_census_256.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("phase2_single_to_512.json"))
    parser.add_argument("--target-ctx", type=int, default=512)
    parser.add_argument("--max-new-loads", type=int)
    parser.add_argument("--max-new-benchmarks", type=int)
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args(argv)
    if args.worker:
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            if not args.model_path or not args.model_id or not args.n_ctx:
                raise ValueError("Argomenti worker mancanti.")
            names = json.loads(args.tensor_names)
            if not isinstance(names, list):
                raise ValueError("Lista tensori malformata.")
            result = _load_worker(
                args.model_path, args.model_id, names, args.n_ctx
            )
        except Exception as exc:
            result = {
                "worker_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("worker_ok") else 1
    try:
        result = run_single_to_512(
            census_path=args.census, output=args.output,
            target_ctx=args.target_ctx,
            max_new_loads=args.max_new_loads,
            max_new_benchmarks=args.max_new_benchmarks,
            benchmark_repeats=args.benchmark_repeats,
            timeout=args.timeout,
            progress=lambda row: print(
                json.dumps(row, ensure_ascii=False), flush=True
            ),
        )
        print(json.dumps({
            "status": result["status"],
            "summary": result["summary"],
            "output": str(args.output),
        }, ensure_ascii=False, indent=2))
        return 0 if result["completed"] else 2
    except Exception as exc:
        print(f"Fase 2 singoli: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
