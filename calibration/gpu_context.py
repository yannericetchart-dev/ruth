"""Measure the diagnostic maximal context with ALL GGUF weights and KV on CUDA.

No offload changes, no quantization changes, no KV-type or runtime tuning.
A loading success alone is insufficient: each isolated worker executes
near the requested context length and verifies native CUDA buffer logs.
Small auxiliary CUDA_Host compute/output buffers may still be required
by llama.cpp and are individually reported, not mistaken for host KV.
The result is a diagnostic reference, NEVER an operational runtime profile.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from hardware import GPUManager
from model import ModelManager
from model.inspect import inspect_gguf

from .baseline_discovery import all_gpu_rules, native_confirmation, minimal_chat_prompt
from .native_memory import parse_native_memory
from .worker_failure import analyze_worker_failure


def verify_gpu_runtime(native: dict) -> dict:
    """Require all KV on CUDA and the main compute buffer on CUDA.

    CUDA_Host/CPU compute and output buffers can exist even with full GPU
    offload. Record them transparently, but they are not evidence that the
    KV cache or GGUF weights have moved into system RAM.
    """
    records = native.get("native_buffer_records") or []
    state = {}
    for category in ("kv", "compute"):
        relevant = [r for r in records if r.get("category") == category]
        gpu = [r for r in relevant if r.get("domain") == "gpu_vram"]
        host = [r for r in relevant if r.get("domain") != "gpu_vram"]
        state[category] = {
            "gpu_buffer_records": len(gpu),
            "other_buffer_records": len(host),
            "gpu_mib_reported": round(sum(r["reported_mib"] for r in gpu), 2),
            "non_gpu_buffers": [
                {"native_buffer": r["native_buffer"], "domain": r["domain"],
                 "reported_mib": r["reported_mib"]}
                for r in host
            ],
            "verified_gpu_resident": bool(gpu) and (
                not host if category == "kv" else True
            ),
        }
    return {
        "verified": all(s["verified_gpu_resident"] for s in state.values()),
        "kv": state["kv"],
        "compute": state["compute"],
        "note": (
            "Tutta la KV deve essere CUDA. Il compute principale deve "
            "comparire su CUDA; piccoli buffer compute CUDA_Host/CPU "
            "possono essere ausiliari e sono esplicitamente riportati. "
            "I log non misurano tutte le allocazioni del driver CUDA."
        ),
    }


def make_context_tokens(llm, architecture: str, n_ctx: int,
                        generated_tokens: int) -> list[int]:
    """Produce a long actual prompt and reserve room for the full decode."""
    if n_ctx < 64 or generated_tokens < 1 or generated_tokens > 32:
        raise ValueError("Contesto o numero di token diagnostici non validi.")
    reserved = generated_tokens + 1
    requested = n_ctx - reserved
    if requested < 16:
        raise ValueError("Contesto insufficiente per la prova.")
    base = minimal_chat_prompt(architecture)
    repetitions = max(1, n_ctx // 4)
    while True:
        prompt = base + (
            " Analizza la sequenza ed esponi il ragionamento con precisione."
            * repetitions
        )
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        if len(tokens) >= requested:
            return list(tokens[:requested])
        repetitions *= 2
        if repetitions > n_ctx * 64:
            raise RuntimeError("Impossibile produrre un prompt della lunghezza richiesta.")


def _worker(model_path: Path, n_ctx: int, generated_tokens: int,
            batch: int = 32) -> dict:
    import pynvml
    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

    if not llama_supports_gpu_offload():
        raise RuntimeError("Backend CUDA non abilitato nel runtime llama-cpp-python.")
    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(
            f"GPU non pronta: {status['errors']}; {status['foreign_processes']}"
        )
    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    architecture = layout["metadata"].get("general.architecture")
    n_blocks = layout["metadata"].get(f"{architecture}.block_count")
    if not isinstance(n_blocks, int) or n_blocks < 1:
        raise RuntimeError("GGUF: numero dei transformer mancante.")
    rules, _ = all_gpu_rules(layout)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (status["gpu"]["uuid"], str(status["gpu"]["index"])):
        raise RuntimeError("Dispositivo CUDA non corrispondente a NVML.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", status["gpu"]["uuid"])

    result = {
        "success": False, "architecture": architecture,
        "n_ctx": n_ctx, "n_gpu_layers": n_blocks + 1,
        "all_gpu_tensor_count": len(layout["tensors"]),
        "offload_kqv": True, "flash_attn": False,
        "kv_type_k": "default_f16", "kv_type_v": "default_f16",
        "n_batch": batch, "n_ubatch": batch,
        "generated_tokens_requested": generated_tokens,
        "measurements": {}, "error": None,
    }
    stop = threading.Event()
    samples = []
    sample_errors = []
    llm = None
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(status["gpu"]["uuid"])

        def used():
            return round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1048576, 1)

        result["measurements"]["vram_before_mib"] = used()

        def sample():
            while not stop.is_set():
                try:
                    samples.append(used())
                except pynvml.NVMLError as exc:
                    sample_errors.append(type(exc).__name__)
                    break
                stop.wait(0.05)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            start = time.perf_counter()
            llm, mapping = LlamaCppPythonMappingAdapter().load(
                model_path, n_gpu_layers=n_blocks + 1, rules=rules,
                n_ctx=n_ctx, n_batch=batch, n_ubatch=batch,
            )
            result["measurements"]["model_load_s"] = round(
                time.perf_counter() - start, 3
            )
            result["measurements"]["vram_after_load_mib"] = used()
            tokens = make_context_tokens(llm, architecture, n_ctx, generated_tokens)
            result["measurements"]["prompt_tokens"] = len(tokens)
            llm.reset()
            t0 = time.perf_counter()
            llm.eval(tokens)
            result["measurements"]["prefill_s"] = round(
                time.perf_counter() - t0, 3
            )
            decoder = llm.generate([], reset=False, temp=0.0, top_k=1)
            generated = 0
            t1 = time.perf_counter()
            try:
                for _ in range(generated_tokens):
                    next(decoder)
                    generated += 1
            finally:
                decoder.close()
            result["measurements"]["decode_s"] = round(
                time.perf_counter() - t1, 3
            )
            result["measurements"]["generated_tokens"] = generated
            result["measurements"]["ending_position_tokens"] = (
                len(tokens) + generated - 1
            )
            result["success"] = generated == generated_tokens
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if llm is not None:
                llm.close()
            stop.set()
            thread.join(timeout=3)
            result["measurements"]["vram_peak_sampled_mib"] = max(
                [result["measurements"]["vram_before_mib"]] + samples
            )
            result["measurements"]["nvml_errors"] = sample_errors
            if sample_errors:
                result["success"] = False
                result["error"] = "Campionamento VRAM NVML incompleto."
    finally:
        pynvml.nvmlShutdown()
    return result


def probe_context(model_path: Path, n_ctx: int,
                  generated_tokens: int = 8, batch: int = 32,
                  timeout: int = 300) -> dict:
    """One new CUDA process per point; no inherited cached CUDA allocations."""
    layout = inspect_gguf(model_path, n_ctx=n_ctx)
    cmd = [
        sys.executable, "-m", "calibration.gpu_context",
        "--worker", "--model-path", str(model_path),
        "--probe-n-ctx", str(n_ctx), "--tokens", str(generated_tokens),
        "--batch", str(batch),
    ]
    try:
        child = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"success": False, "failure_class": "timeout",
                "n_ctx": n_ctx, "error": f"Timeout dopo {timeout}s"}
    try:
        report = json.loads(child.stdout)
    except (ValueError, TypeError):
        report = {"success": False, "error": "JSON worker non valido."}
    report["worker_returncode"] = child.returncode
    if child.returncode != 0:
        report["success"] = False
    diagnosis = analyze_worker_failure(
        child.stderr, report.get("error"),
        child.returncode, report.get("success") is True,
    )
    report.update(diagnosis)
    native = parse_native_memory(child.stderr)
    report["native_memory"] = native
    runtime = verify_gpu_runtime(native)
    report["gpu_runtime_validation"] = runtime
    placement = native_confirmation(layout, child.stderr)
    report["full_gpu_native_verification"] = {
        "verified": placement["verified"],
        "tensor_count": placement["tensor_count"],
        "unverified_names": placement["unverified_names"][:20],
        "native_logged_occurrences": placement["native_override"][
            "native_logged_occurrence_count"
        ],
    }
    measured = report.get("measurements") or {}
    valid = (
        report.get("success") is True
        and placement["verified"]
        and runtime["verified"]
        and report.get("n_ctx") == n_ctx
        and measured.get("generated_tokens") == generated_tokens
        and measured.get("prompt_tokens") == n_ctx - generated_tokens - 1
        and isinstance(measured.get("vram_peak_sampled_mib"), (int, float))
        and math.isfinite(measured["vram_peak_sampled_mib"])
        and measured["vram_peak_sampled_mib"] > 0
        and not measured.get("nvml_errors")
    )
    report["success"] = valid
    if valid:
        report["failure_class"] = None
    elif report.get("suspected_cuda_oom") is True:
        report["failure_class"] = "native_cuda_oom"
    elif report.get("failure_hint") == "ambiguous_context_initialization":
        report["failure_class"] = "ambiguous_context_initialization"
    elif not placement["verified"]:
        report["failure_class"] = "native_weights_not_verified"
    elif not runtime["verified"]:
        report["failure_class"] = "runtime_not_proven_gpu_resident"
    else:
        report["failure_class"] = "inference_or_worker_error"
    return report


def context_limit(layout: dict) -> int:
    arch = layout["metadata"].get("general.architecture")
    maximum = layout["metadata"].get(f"{arch}.context_length")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 64:
        raise ValueError("Lunghezza contesto dichiarata dal GGUF assente/non valida.")
    return maximum


def search_gpu_context(
    model_id: str, *, max_ctx: int | None = None,
    min_ctx: int = 64, tokens: int = 8,
    batch: int = 32, timeout: int = 300,
    confirm_rounds: int = 2,
    retry_ambiguous_context: bool = False,
    root: str | Path | None = None,
    baseline_selection: dict | None = None,
    persist_reference: bool = True,
    progress=None,
) -> dict:
    """Bounded doubling + token-granularity bisection, then confirmation.

    Failure modes other than a native OOM are not assumed to be memory
    capacity limits. Inconclusive results stop the search unless explicitly
    opted into treating ambiguous context errors as a search boundary.
    """
    if model_id not in ModelManager.CATALOG:
        raise ValueError(f"Modello non supportato: {model_id}")
    if (not isinstance(tokens, int) or not 1 <= tokens <= 32
        or not isinstance(batch, int) or batch < 1
        or not isinstance(min_ctx, int) or min_ctx < 64
        or not isinstance(confirm_rounds, int) or confirm_rounds < 1
        or not isinstance(timeout, int) or timeout < 1):
        raise ValueError("Parametri diagnostici non validi.")
    if max_ctx is not None and (not isinstance(max_ctx, int) or max_ctx < min_ctx):
        raise ValueError("max_ctx non valido.")
    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(f"GPU non pronta: {status['errors']}")
    manager = ModelManager(root=root)
    selection = baseline_selection if baseline_selection is not None else manager.selected_baseline(model_id)
    if selection is None:
        raise RuntimeError(
            f"Baseline full-GPU non selezionata per {model_id}. "
            "Eseguire prima calibration.baseline_discovery."
        )
    if (selection.get("native_verified_all_gpu") is not True
        or selection.get("model_id", model_id) != model_id
        or selection.get("repository", ModelManager.CATALOG[model_id]["repository"])
           != ModelManager.CATALOG[model_id]["repository"]):
        raise RuntimeError("Baseline non verificata o di un'altra configurazione.")
    model = Path(selection["local_path"])
    if not model.is_file() or model.name != selection.get("filename"):
        raise RuntimeError("GGUF selezionato assente o nome incoerente.")
    layout = inspect_gguf(model, n_ctx=min_ctx)
    arch = layout["metadata"].get("general.architecture")
    if arch != selection["architecture"]:
        raise RuntimeError("Architettura manifest diversa dal GGUF selezionato.")
    model_max = context_limit(layout)
    ceiling = min(model_max, max_ctx) if max_ctx is not None else model_max
    if ceiling < min_ctx:
        raise ValueError("Contesto minimo oltre il massimo del modello.")
    total_mib = status["memory"]["total_mib"]
    result = {
        "phase": "full_gpu_resident_reference_context",
        "model_id": model_id,
        "quantization": selection["quantization"],
        "filename": selection["filename"],
        "architecture": arch,
        "gpu_total_mib": total_mib,
        "n_ctx_min": min_ctx,
        "model_declared_context_length": model_max,
        "search_ceiling": ceiling,
        "tokens_per_probe": tokens,
        "batch": batch,
        "confirmed_max_gpu_resident_n_ctx": None,
        "profile_generated": False,
        "selected_operational_n_ctx": None,
        "attempts": [],
        "success": False,
        "measurement_completed": False,
        "note": (
            "Contesto diagnostico: mapping di tutti i pesi GPU invariato, "
            "KV in GPU e compute CUDA principale verificati da log, prefill vicino al "
            "contesto dichiarato. Nessun margine VRAM operativo implicito, "
            "nessuna modifica a quantizzazione KV o parametri llama.cpp."
        ),
    }

    def measure(ctx: int) -> dict:
        if progress is not None:
            progress({"event": "context_probe_start", "n_ctx": ctx,
                      "model": model_id, "token_count": tokens})
        candidate = probe_context(
            model, ctx, generated_tokens=tokens, batch=min(batch, ctx),
            timeout=timeout,
        )
        candidate["requested_n_ctx"] = ctx
        result["attempts"].append(candidate)
        if progress is not None:
            progress({"event": "context_probe_result", "n_ctx": ctx,
                      "model": model_id,
                      "success": candidate.get("success") is True,
                      "failure_class": candidate.get("failure_class"),
                      "peak_mib": (candidate.get("measurements") or {}).get(
                          "vram_peak_sampled_mib"),
                      "native_gpu_weights": (candidate.get(
                          "full_gpu_native_verification") or {}).get("verified"),
                      "native_gpu_kv": (candidate.get(
                          "gpu_runtime_validation") or {}).get("kv", {}).get(
                              "verified_gpu_resident")})
        return candidate

    def is_boundary(failure: dict) -> bool:
        cause = failure.get("failure_class")
        if cause == "native_cuda_oom":
            return True
        if cause == "ambiguous_context_initialization" and retry_ambiguous_context:
            result["ambiguous_boundary_used"] = True
            return True
        result["error"] = (
            f"Prova inconclusiva a n_ctx={failure.get('requested_n_ctx')}: "
            f"{cause}, {failure.get('error')}. "
            "Non interpretata come limite di memoria."
        )
        return False

    low = min_ctx
    first = measure(low)
    if not first["success"]:
        result["error"] = (
            f"Nessun contesto GPU-resident verificato a {min_ctx}: "
            f"{first.get('failure_class')}; {first.get('error')}"
        )
        return result

    high = None
    while low < ceiling:
        ctx = min(ceiling, low * 2)
        probe = measure(ctx)
        if probe["success"]:
            low = ctx
            continue
        if not is_boundary(probe):
            return result
        high = ctx
        break

    if high is not None:
        while high - low > 1:
            mid = low + (high - low) // 2
            probe = measure(mid)
            if probe["success"]:
                low = mid
            elif is_boundary(probe):
                high = mid
            else:
                return result

    result["candidate_max_gpu_resident_n_ctx"] = low
    if progress is not None:
        progress({"event": "context_candidate_limit", "n_ctx": low,
                  "failed_boundary": high, "model": model_id})
    result["confirmation"] = []
    for confirmation_index in range(1, confirm_rounds + 1):
        if progress is not None:
            progress({"event": "context_confirmation", "n_ctx": low,
                      "round": confirmation_index, "total": confirm_rounds,
                      "model": model_id})
        confirmation = measure(low)
        result["confirmation"].append({
            "success": confirmation["success"],
            "failure_class": confirmation.get("failure_class"),
            "vram_peak_sampled_mib": (confirmation.get("measurements") or {}).get(
                "vram_peak_sampled_mib"
            ),
        })
        if not confirmation["success"]:
            result["error"] = (
                f"Ripetizione di conferma fallita a n_ctx={low}. "
                "Nessun valore massimo confermato."
            )
            return result

    result["confirmed_max_gpu_resident_n_ctx"] = low
    result["success"] = True
    result["measurement_completed"] = True
    result["highest_context_tested_and_failed"] = high
    result["search_reached_declared_limit"] = low == model_max
    result["confirmation_min_sampled_headroom_mib"] = round(
        min(total_mib - row["vram_peak_sampled_mib"]
            for row in result["confirmation"]), 1
    )
    result["reference_is_operationally_safe"] = False
    if not persist_reference:
        return result
    target = model.parent / "gpu_context_reference.json"
    temp = model.parent / "gpu_context_reference.json.tmp"
    temp.write_text(json.dumps({
        "model_id": model_id,
        "filename": selection["filename"],
        "quantization": selection["quantization"],
        "n_ctx": low,
        "gpu_total_mib": total_mib,
        "confirmation": result["confirmation"],
        "profile_generated": False,
        "operationally_safe": False,
    }, indent=2) + "\n")
    os.replace(temp, target)
    result["reference_manifest"] = str(target)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Max n_ctx diagnostico con tutti i pesi, KV e compute GPU"
    )
    parser.add_argument("--model", choices=sorted(ModelManager.CATALOG),
                        default="gemma-2-9b-it")
    parser.add_argument("--source", default="bartowski")
    parser.add_argument("--engine", default="llama.cpp")
    parser.add_argument("--method", default="llama_cpp_python")
    parser.add_argument("--max-ctx", type=int)
    parser.add_argument("--min-ctx", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--confirm-rounds", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retry-ambiguous-context", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", help=argparse.SUPPRESS)
    parser.add_argument("--probe-n-ctx", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            output = _worker(
                Path(args.model_path), args.probe_n_ctx,
                generated_tokens=args.tokens, batch=args.batch,
            )
        except Exception as exc:
            output = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            from .profile_identity import validate_choices
            validate_choices(args.model, args.source, args.engine, args.method)
            output = search_gpu_context(
                args.model, max_ctx=args.max_ctx, min_ctx=args.min_ctx,
                tokens=args.tokens, batch=args.batch,
                timeout=args.timeout, confirm_rounds=args.confirm_rounds,
                retry_ambiguous_context=args.retry_ambiguous_context,
            )
        except Exception as exc:
            output = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not output.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
