"""Automatic, quality-first, full-GPU GGUF baseline discovery.

The first candidate with a native-verified *all-tensor GPU* placement,
successful minimal inference and reliable NVML data becomes the baseline.
No partial CPU weight offload, no context optimization, and no automatic
retuning are permitted in this stage. Skipped files remain valid candidates
for future tuned profiles.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from hardware import GPUManager
from model import ModelManager
from model.inspect import inspect_gguf

from .gguf_catalog import (
    REPOSITORY, gguf_payload_bytes, inventory, remote_header,
)
from .mapping_native_validation import verify_override_logs
from .native_memory import parse_native_memory
from .worker_failure import analyze_worker_failure


def all_gpu_rules(layout: dict) -> tuple[list[dict], dict]:
    """Generate verified, non-overlapping rules covering every GGUF tensor.

    Gemma 2's tied token embedding is isolated and must appear twice
    in native logs. Llama 3.1 has an independent output.weight tensor:
    its embedding and output are verified independently on CUDA.
    Llama rope_freqs.weight appears once in the GGUF inventory, but
    once per transformer block in the native placement logs.
    """
    names = [t["name"] for t in layout["tensors"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Inventario GGUF vuoto o duplicato.")
    architecture = layout["metadata"].get("general.architecture")
    if architecture not in ("gemma2", "llama", "qwen2"):
        raise ValueError(f"Architettura non validata per full-GPU: {architecture}")
    tied = architecture == "gemma2" and "token_embd.weight" in names
    if architecture in ("llama", "qwen2") and not {"token_embd.weight", "output.weight"}.issubset(names):
        raise ValueError(f"{architecture}: embedding e output.weight indipendenti mancanti.")
    grouped = [n for n in names if re.fullmatch(r"blk\.\d+\..+", n)]
    other = [n for n in names if n not in grouped and (n != "token_embd.weight" or not tied)]
    rules = []
    if grouped:
        rules.append({"pattern": r"^blk\.\d+\..+$", "device": "gpu"})
    for name in sorted(other):
        rules.append({"pattern": "^" + re.escape(name) + "$", "device": "gpu"})
    if tied:
        rules.append({"pattern": r"^(token_embd\.weight)$", "device": "gpu"})

    resolved = []
    for rule in rules:
        matches = [n for n in names if re.search(rule["pattern"], n)]
        resolved.extend(matches)
    if sorted(resolved) != sorted(names):
        raise RuntimeError("Copertura nativa di tutti i tensori incompleta.")

    occurrences = {"token_embd.weight": 2} if tied else {}
    if architecture in ("llama", "qwen2") and "rope_freqs.weight" in names:
        # llama.cpp materializes the one GGUF rope_freqs.weight tensor for
        # every transformer block. On Llama 3.1 8B it logs 32 native
        # override occurrences, all of which must use GPU VRAM.
        block_count = layout["metadata"].get(f"{architecture}.block_count")
        if (not isinstance(block_count, int) or isinstance(block_count, bool)
            or block_count < 1):
            raise ValueError(f"{architecture}: block_count non valido per rope_freqs.weight.")
        occurrences["rope_freqs.weight"] = block_count

    expected = {
        "host_tensor_names": [],
        "gpu_tensor_names": names,
        "expected_native_log_occurrences": occurrences,
        "output_weight_target": "gpu",
    }
    return rules, expected


def native_confirmation(layout: dict, stderr: str) -> dict:
    _, expected = all_gpu_rules(layout)
    check = verify_override_logs(expected, stderr)
    return {
        "verified": check["verified"],
        "tensor_count": len(layout["tensors"]),
        "unverified_names": [
            entry["name"] for entry in check["tensors"]
            if not entry["verified_by_log"]
        ],
        "tied_embedding_output_verified": (
            check.get("output_placement_verified") if
            "token_embd.weight" in expected["expected_native_log_occurrences"]
            else None
        ),
        "native_override": check,
    }


def minimal_chat_prompt(architecture: str) -> str:
    """Architecture-specific chat formatting with add_bos=True tokenization."""
    if architecture == "gemma2":
        return "<start_of_turn>user\nCiao.\n<end_of_turn>\n<start_of_turn>model\n"
    if architecture == "llama":
        # Llama tokenizer prepends BOS via add_bos=True, so do not put
        # <|begin_of_text|> in the prompt and duplicate it.
        return (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Ciao.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if architecture == "qwen2":
        return "<|im_start|>user\nCiao.<|im_end|>\n<|im_start|>assistant\n"
    raise ValueError(f"Chat template non validato: {architecture}")


def _gpu_worker(path: Path, n_ctx: int = 64, n_batch: int = 32) -> dict:
    """Runs in its own short-lived process; never persists a profile."""
    import pynvml as nvml
    from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter
    from llama_cpp.llama_cpp import llama_supports_gpu_offload

    if not llama_supports_gpu_offload():
        raise RuntimeError("Runtime llama-cpp-python privo di backend CUDA.")
    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(f"GPU non libera: {status['errors']}; {status['foreign_processes']}")
    layout = inspect_gguf(path, n_ctx=n_ctx)
    arch = layout["metadata"].get("general.architecture")
    blocks = layout["metadata"].get(f"{arch}.block_count")
    if not isinstance(blocks, int) or blocks <= 0:
        raise ValueError(f"Numero di blocchi non valido per {arch}.")
    rules, _ = all_gpu_rules(layout)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (status["gpu"]["uuid"], str(status["gpu"]["index"])):
        raise RuntimeError("CUDA_VISIBLE_DEVICES non coincide con la GPU NVML selezionata.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", status["gpu"]["uuid"])
    report = {
        "success": False, "file": path.name,
        "architecture": arch,
        "n_ctx_minimal_probe": n_ctx,
        "n_gpu_layers": blocks + 1,
        "requested_gpu_tensors": len(layout["tensors"]),
        "model_weights_bytes": layout["weights_bytes"],
        "n_batch": n_batch,
        "n_ubatch": n_batch,
        "measurements": {},
        "error": None,
    }
    memory = report["measurements"]
    samples = []
    sampling_error = []
    stop = threading.Event()
    llm = None
    nvml.nvmlInit()
    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(status["gpu"]["uuid"])
        mib = 1024 * 1024

        def usage():
            return round(nvml.nvmlDeviceGetMemoryInfo(handle).used / mib, 1)

        memory["vram_before_mib"] = usage()

        def sample():
            while not stop.is_set():
                try:
                    samples.append(usage())
                except nvml.NVMLError as exc:
                    sampling_error.append(str(exc))
                    break
                stop.wait(0.05)

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            start = time.perf_counter()
            llm, mapping = LlamaCppPythonMappingAdapter().load(
                path, n_gpu_layers=blocks + 1,
                rules=rules, n_ctx=n_ctx,
                n_batch=n_batch, n_ubatch=n_batch,
            )
            memory["load_seconds"] = round(time.perf_counter() - start, 3)
            memory["vram_after_load_mib"] = usage()
            # Loading alone is insufficient; verify the context can execute.
            prompt = minimal_chat_prompt(arch)
            tokens = llm.tokenize(prompt.encode(), add_bos=True, special=True)
            if len(tokens) + 2 >= n_ctx:
                raise RuntimeError("Probe minimale: prompt troppo lungo.")
            llm.reset()
            llm.eval(tokens)
            generator = llm.generate([], reset=False, temp=0.0, top_k=1)
            try:
                next(generator)
                next(generator)
            finally:
                generator.close()
            memory["minimal_inference_completed"] = True
            report["success"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if llm is not None:
                llm.close()
            stop.set()
            thread.join(timeout=2)
            memory["vram_peak_sampled_mib"] = max(
                [memory["vram_before_mib"]] + samples
            )
            memory["vram_sampling_error"] = sampling_error
            if sampling_error:
                report["success"] = False
                report["error"] = "Campionamento NVML incompleto."
    finally:
        nvml.nvmlShutdown()
    return report


def probe_gpu_candidate(path: Path, n_ctx: int = 64,
                        n_batch: int = 32, timeout: int = 180) -> dict:
    """Native-verify a candidate in an isolated worker; record failure logs."""
    layout = inspect_gguf(path, n_ctx=n_ctx)
    command = [
        sys.executable, "-m", "calibration.baseline_discovery",
        "--worker", "--model-path", str(path.resolve()),
        "--probe-n-ctx", str(n_ctx), "--batch", str(n_batch),
    ]
    try:
        child = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "success": False, "reason": "timeout",
            "error": f"Worker scaduto dopo {timeout}s",
        }
    try:
        report = json.loads(child.stdout)
    except (json.JSONDecodeError, TypeError):
        report = {"success": False, "error": "Worker non ha prodotto JSON valido."}
    report["worker_returncode"] = child.returncode
    if child.returncode != 0:
        report["success"] = False
    diagnostic = analyze_worker_failure(
        child.stderr, report.get("error"),
        child.returncode, report.get("success") is True,
    )
    report.update(diagnostic)
    report["native_memory"] = parse_native_memory(child.stderr)
    # This is an all-GPU search: every tensor requires positive native
    # evidence, including both physical uses of the Gemma 2 tied weight.
    verified = native_confirmation(layout, child.stderr)
    report["full_gpu_native_verification"] = verified
    if not verified["verified"]:
        report["success"] = False
        report["error"] = report.get("error") or "Tensori non tutti confermati su CUDA."
    return report


def choose_baseline(
    root: Path | None = None, repo_id: str | None = None,
    probe_n_ctx: int = 64, n_batch: int = 32,
    timeout: int = 180, inventory_only: bool = False,
    model_id: str = "gemma-2-9b-it",
    progress=None, persist_selection: bool = True,
) -> dict:
    if (not isinstance(probe_n_ctx, int) or probe_n_ctx < 32
        or not isinstance(n_batch, int) or not 1 <= n_batch <= probe_n_ctx
        or not isinstance(timeout, int) or timeout < 1):
        raise ValueError("Probe minimo: ctx>=32, 1<=batch<=ctx, timeout>0.")

    from huggingface_hub import hf_hub_download

    spec = ModelManager.CATALOG.get(model_id)
    if spec is None:
        raise ValueError(f"Modello non registrato: {model_id}")
    if repo_id is None:
        repo_id = spec["repository"]
    if repo_id != spec["repository"]:
        raise ValueError(
            "Il repository non corrisponde al modello selezionato: "
            f"{model_id} -> {spec['repository']}"
        )

    gpu = GPUManager.calibration_status()
    if not gpu["ready"]:
        raise RuntimeError(f"GPU non pronta: {gpu['errors']}; {gpu['foreign_processes']}")
    vram_bytes = int(gpu["memory"]["total_mib"] * 1024 * 1024)
    model_dir = (
        Path(root).expanduser().resolve() if root is not None
        else ModelManager().root / model_id
    )
    models = inventory(repo_id)
    unranked = [
        c["filename"] for c in models if c.get("quality_priority") is None
    ]
    report = {
        "phase": "quality_first_full_gpu_baseline",
        "model_id": model_id,
        "expected_architecture": spec["architecture"],
        "repository": repo_id,
        "gpu": gpu["gpu"]["name"],
        "total_vram_mib": gpu["memory"]["total_mib"],
        "probe_n_ctx": probe_n_ctx,
        "probe_batch": n_batch,
        "rank_is_empirical_prior_not_measured_quality": True,
        "inventory": models,
        "unranked_quantizations": unranked,
        "attempts": [],
        "baseline": None,
        "success": False,
        "profile_generated": False,
        "next_phase_unlocked": False,
        "note": (
            "Un GGUF escluso per pesi serializzati maggiori della VRAM "
            "non e' escluso dai profili CPU/GPU con tuning successivi. "
            "Nessun fallback implicito su CPU per i pesi. "
            "Tutti i tensori, incluso embedding/output condiviso, devono "
            "avere conferma nativa CUDA."
        ),
    }
    if progress is not None:
        progress({"event": "inventory_ready", "model": model_id,
                  "count": len(models), "unranked": len(unranked)})
    if inventory_only:
        report["inventory_only"] = True
        return report

    if unranked:
        # Unknown formats may be superior to currently ranked candidates.
        # Inventory is still returned, but selection must not silently
        # skip them or call a lower-ranked model the best available.
        report["error"] = (
            "Quantizzazioni non classificabili rilevate. Inventario completo "
            "disponibile; aggiornare la gerarchia di qualita' prima della "
            "selezione della baseline."
        )
        return report

    for candidate in models:
        record = {
            "filename": candidate["filename"],
            "quantization": candidate["quantization"],
            "quality_priority": candidate["quality_priority"],
            "status": None,
        }
        report["attempts"].append(record)
        if progress is not None:
            progress({"event": "candidate_start", "model": model_id,
                      "quantization": candidate["quantization"],
                      "filename": candidate["filename"]})
        if candidate.get("cuda_baseline_eligible") is False:
            record["status"] = "excluded_publisher_arm_prepacked_format"
            record["future_cpu_gpu_tuning_allowed"] = True
            if progress is not None:
                progress({"event": "candidate_decision", "model": model_id,
                          "quantization": candidate["quantization"],
                          "status": record["status"]})
            continue

        local = model_dir / candidate["filename"]
        if local.is_file():
            layout = inspect_gguf(local, n_ctx=probe_n_ctx)
            exact = layout["weights_bytes"]
            record["weight_size_source"] = "local_gguf_tensors"
        else:
            # A tiny ranged-header read, *not* a GGUF download.
            try:
                header = remote_header(
                    repo_id, candidate["filename"], candidate["file_bytes"]
                )
                exact = header["exact_serialized_tensor_bytes"]
                record["weight_size_source"] = "remote_gguf_tensor_header"
            except Exception as exc:
                # No guessed exclusion and no multi-gigabyte download
                # without verified tensor weights. Stop to avoid losing a
                # higher-quality eligible candidate silently.
                record["status"] = "unverified_remote_header"
                record["error"] = f"{type(exc).__name__}: {exc}"
                report["error"] = "Inventario peso GGUF non verificabile."
                return report

        record["exact_serialized_weight_mib"] = round(exact / 1048576, 2)
        if exact > vram_bytes:
            record["status"] = "excluded_weight_payload_exceeds_total_vram"
            record["future_cpu_gpu_tuning_allowed"] = True
            if progress is not None:
                progress({"event": "candidate_decision", "model": model_id,
                          "quantization": candidate["quantization"],
                          "status": record["status"],
                          "weight_mib": record["exact_serialized_weight_mib"]})
            continue

        if not local.is_file():
            if progress is not None:
                progress({"event": "candidate_download", "model": model_id,
                          "quantization": candidate["quantization"],
                          "filename": candidate["filename"]})
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
                local = Path(hf_hub_download(
                    repo_id=repo_id, filename=candidate["filename"],
                    local_dir=str(model_dir),
                ))
            except Exception as exc:
                record["status"] = "download_failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                report["error"] = "Download fallito: impossibile verificare questo candidato."
                return report

        if progress is not None:
            progress({"event": "candidate_gpu_probe", "model": model_id,
                      "quantization": candidate["quantization"],
                      "n_ctx": probe_n_ctx, "batch": n_batch})
        try:
            measured = probe_gpu_candidate(
                local, n_ctx=probe_n_ctx,
                n_batch=n_batch, timeout=timeout,
            )
        except Exception as exc:
            measured = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["test"] = measured
        fully_verified = (
            measured.get("success") is True
            and (measured.get("full_gpu_native_verification") or {}).get("verified") is True
            and (measured.get("measurements") or {}).get("minimal_inference_completed") is True
            and measured.get("architecture") == spec["architecture"]
        )
        if not fully_verified:
            record["status"] = "failed_full_gpu_native_probe"
            record["future_cpu_gpu_tuning_allowed"] = True
            if progress is not None:
                progress({"event": "candidate_decision", "model": model_id,
                          "quantization": candidate["quantization"],
                          "status": record["status"],
                          "failure_hint": measured.get("failure_hint"),
                          "error": measured.get("error")})
            # Stop on diagnostic errors not caused by model feasibility:
            # don't mislabel broken adapter or missing logs as a bad GGUF.
            if measured.get("failure_hint") not in (
                "explicit_cuda_oom_evidence",
                "ambiguous_context_initialization",
            ) and measured.get("suspected_cuda_oom") is not True:
                record["needs_diagnostic_review"] = True
                report["error"] = (
                    "Probe GPU inconclusivo: errore non classificato. "
                    "Non saltare silenziosamente un candidato di qualita' superiore."
                )
                return report
            continue

        record["status"] = "selected_full_gpu_baseline"
        report["baseline"] = {
            "repository": repo_id,
            "model_id": model_id,
            "architecture": spec["architecture"],
            "filename": candidate["filename"],
            "quantization": candidate["quantization"],
            "local_path": str(local),
            "native_verified_all_gpu": True,
            "baseline_probe_n_ctx": probe_n_ctx,
            "baseline_probe_batch": n_batch,
            "max_gpu_resident_n_ctx": None,
        }
        report["success"] = True
        report["next_phase_unlocked"] = True
        if progress is not None:
            progress({"event": "candidate_decision", "model": model_id,
                      "quantization": candidate["quantization"],
                      "status": record["status"],
                      "peak_mib": (measured.get("measurements") or {}).get(
                          "vram_peak_sampled_mib")})
        if not persist_selection:
            break
        # Atomic persistent selection: no update on failure, no switch to
        # a new operational runtime profile without a separate step.
        target = model_dir / "baseline_selection.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report["baseline"], indent=2) + "\n")
        os.replace(temp, target)
        report["baseline_manifest"] = str(target)
        break
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Discover Bartowski GGUFs and select first native-verified full-GPU baseline"
    )
    parser.add_argument("--model", choices=sorted(ModelManager.CATALOG),
                        default="gemma-2-9b-it",
                        help="Modello nel catalogo Ruth (baseline separata)")
    parser.add_argument("--source", default="bartowski")
    parser.add_argument("--engine", default="llama.cpp")
    parser.add_argument("--method", default="llama_cpp_python")
    parser.add_argument("--repository", default=None,
                        help="Opzionale, deve coincidere col repository del modello")
    parser.add_argument("--root", default=None, help="Cartella modelli (non cambia profili)")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--probe-n-ctx", type=int, default=64)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model-path", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            result = _gpu_worker(
                Path(args.model_path), n_ctx=args.probe_n_ctx,
                n_batch=args.batch,
            )
        except Exception as exc:
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        try:
            from .profile_identity import validate_choices
            validate_choices(args.model, args.source, args.engine, args.method)
            result = choose_baseline(
                root=Path(args.root) if args.root else None,
                repo_id=args.repository,
                probe_n_ctx=args.probe_n_ctx,
                n_batch=args.batch,
                timeout=args.timeout,
                inventory_only=args.inventory_only,
                model_id=args.model,
            )
        except Exception as exc:
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.inventory_only and not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
