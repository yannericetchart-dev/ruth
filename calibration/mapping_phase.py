"""Phase 2 pilot: native CPU/GPU GGUF weight placement verification.

Two intentionally bounded experiments per verified baseline:
1. move transformer block 0 to host, retain every other GGUF tensor on CUDA;
2. move one small attention tensor in the last block to host, keep the
   remaining weights on CUDA.
This is evidence collection, not mapping optimization, context tuning,
or a production profile. All workers are isolated and never persist model
configuration changes. The original phase-1 results are read-only.
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys
import threading
import time

from hardware import GPUManager
from model.inspect import inspect_gguf

from .baseline_discovery import all_gpu_rules, minimal_chat_prompt
from .gpu_context import verify_gpu_runtime
from .mapping_native_validation import verify_override_logs
from .native_memory import parse_native_memory
from .worker_failure import analyze_worker_failure


SCHEMA_VERSION = 1
PROBE_N_CTX = 64
PROBE_BATCH = 32
PROBE_TOKENS = 2
MODES = ("block0_host", "attention_tensor_host")


def native_mapping_trials(layout: dict) -> list[dict]:
    """Build fully covered, non-overlapping exact native GGUF placements."""
    metadata = layout.get("metadata") or {}
    arch = metadata.get("general.architecture")
    blocks = metadata.get(f"{arch}.block_count")
    if (not isinstance(blocks, int) or isinstance(blocks, bool)
        or blocks < 1):
        raise ValueError("Mapping: numero di blocchi GGUF non valido.")

    inventory = layout.get("tensors") or []
    all_names = [t.get("name") for t in inventory]
    if not all_names or len(all_names) != len(set(all_names)):
        raise ValueError("Mapping: inventario GGUF duplicato o vuoto.")
    by_block = {
        n: sorted(
            t["name"] for t in inventory
            if t.get("block") == n
            and isinstance(t.get("name"), str)
            and t["name"].startswith(f"blk.{n}.")
        )
        for n in range(blocks)
    }
    if not all(by_block.values()):
        raise ValueError("Mapping: almeno un blocco GGUF privo di tensori.")
    _, expected_all_gpu = all_gpu_rules(layout)
    selected = [
        t for t in inventory
        if isinstance(t.get("name"), str)
        and t["name"] in by_block[blocks - 1]
        and ".attn_" in t["name"]
        and isinstance(t.get("bytes"), int)
        and t["bytes"] > 0
    ]
    if not selected:
        raise ValueError("Mapping: nessun tensore attention nell'ultimo blocco.")
    smallest = min(selected, key=lambda t: (t["bytes"], t["name"]))

    def build(mode: str, host_set: set[str]) -> dict:
        if not host_set or not host_set.issubset(all_names):
            raise ValueError("Mapping: host tensor list non valida.")
        rules = []
        for block, names in by_block.items():
            host = [n for n in names if n in host_set]
            gpu = [n for n in names if n not in host_set]
            if len(host) == len(names):
                rules.append({
                    "pattern": rf"^blk\\.{block}\\..+$",
                    "device": "cpu",
                })
            elif not host:
                rules.append({
                    "pattern": rf"^blk\\.{block}\\..+$",
                    "device": "gpu",
                })
            else:
                for name in host:
                    rules.append({
                        "pattern": "^" + re.escape(name) + "$",
                        "device": "cpu",
                    })
                for name in gpu:
                    rules.append({
                        "pattern": "^" + re.escape(name) + "$",
                        "device": "gpu",
                    })
        assigned_block = {name for names in by_block.values() for name in names}
        for name in sorted(set(all_names) - assigned_block):
            pattern = (
                r"^(token_embd\\.weight)$"
                if arch == "gemma2" and name == "token_embd.weight"
                else "^" + re.escape(name) + "$"
            )
            rules.append({"pattern": pattern, "device": "gpu"})

        matches = {}
        for rule in rules:
            names = [n for n in all_names if re.search(rule["pattern"], n)]
            if not names:
                raise RuntimeError("Mapping: regex senza tensori: " + rule["pattern"])
            for name in names:
                if name in matches:
                    raise RuntimeError("Mapping: tensor regex sovrapposte: " + name)
                matches[name] = rule["device"]
        if set(matches) != set(all_names):
            raise RuntimeError("Mapping: copertura tensori incompleta.")
        expected_host = sorted(n for n, device in matches.items()
                               if device == "cpu")
        expected_gpu = sorted(n for n, device in matches.items()
                              if device == "gpu")
        if expected_host != sorted(host_set):
            raise RuntimeError("Mapping: piano host non coerente.")
        return {
            "id": mode, "rules": rules,
            "host_tensor_names": expected_host,
            "gpu_tensor_names": expected_gpu,
            "expected_native_log_occurrences":
                expected_all_gpu["expected_native_log_occurrences"],
            "output_weight_target": "gpu",
            "host_tensor_count": len(expected_host),
            "gpu_tensor_count": len(expected_gpu),
            "host_gguf_bytes": sum(
                t["bytes"] for t in inventory if t["name"] in host_set
            ),
            "all_tensor_count": len(all_names),
            "block_count": blocks,
        }

    return [
        build("block0_host", set(by_block[0])),
        build("attention_tensor_host", {smallest["name"]}),
    ]


def _worker(model_path: Path, model_id: str, mode: str) -> dict:
    """Isolated native worker; result is JSON, DEBUG placement stays stderr."""
    import pynvml
    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

    if not llama_supports_gpu_offload():
        raise RuntimeError("Mapping: backend CUDA llama.cpp non disponibile.")
    status = GPUManager.calibration_status()
    if not status.get("ready"):
        raise RuntimeError("Mapping: GPU non pronta.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu = status["gpu"]
    if visible and visible not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError("Mapping: CUDA_VISIBLE_DEVICES diversa dalla GPU.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    layout = inspect_gguf(model_path, n_ctx=PROBE_N_CTX)
    trial = next(
        (t for t in native_mapping_trials(layout) if t["id"] == mode),
        None,
    )
    if trial is None:
        raise ValueError("Mapping: esperimento sconosciuto.")
    arch = layout["metadata"]["general.architecture"]
    report = {
        "worker_ok": False, "mode": mode, "model_id": model_id,
        "n_ctx": PROBE_N_CTX, "n_batch": PROBE_BATCH,
        "n_gpu_layers": trial["block_count"] + 1,
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
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])
        def usage():
            return round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1048576, 1)
        report["measurements"]["vram_before_mib"] = usage()

        def monitor():
            while not stop.is_set():
                try:
                    samples.append(usage())
                except pynvml.NVMLError as exc:
                    errors.append(type(exc).__name__)
                    break
                stop.wait(0.05)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            start = time.perf_counter()
            llm, applied = LlamaCppPythonMappingAdapter().load(
                model_path,
                n_gpu_layers=trial["block_count"] + 1,
                rules=trial["rules"], n_ctx=PROBE_N_CTX,
                n_batch=PROBE_BATCH, n_ubatch=PROBE_BATCH,
            )
            report["measurements"]["load_s"] = round(
                time.perf_counter() - start, 3
            )
            report["measurements"]["vram_after_load_mib"] = usage()
            prompt = minimal_chat_prompt(arch, model_id)
            tokens = llm.tokenize(prompt.encode(), add_bos=True, special=True)
            if len(tokens) + PROBE_TOKENS + 1 >= PROBE_N_CTX:
                raise RuntimeError("Mapping: prompt oltre il contesto diagnostico.")
            llm.reset()
            start = time.perf_counter()
            llm.eval(tokens)
            report["measurements"]["prefill_s"] = round(
                time.perf_counter() - start, 3
            )
            generated = llm.generate([], reset=False, temp=0.0, top_k=1)
            start = time.perf_counter()
            completed = 0
            try:
                for _ in range(PROBE_TOKENS):
                    next(generated)
                    completed += 1
            finally:
                generated.close()
            report["measurements"]["decode_s"] = round(
                time.perf_counter() - start, 3
            )
            report["measurements"]["generated_tokens"] = completed
            report["worker_ok"] = completed == PROBE_TOKENS
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
    if mode := trial.get("id") not in MODES:
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
