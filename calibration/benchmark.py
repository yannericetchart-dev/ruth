"""Misura isolata di prefill, decode e VRAM per un mapping CUDA.

Solo benchmark diagnostico del modello: n_ctx=512 e' un contesto
tecnico di prova, NON il contesto operativo del futuro profilo.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from importlib.metadata import version

from hardware import GPUManager
from model import ModelManager


def _worker(gpu_layers: int, max_tokens: int, cpu_tensor: str | None = None) -> dict:
    import pynvml as nvml

    if gpu_layers < 1 or max_tokens < 64 or max_tokens > 256:
        raise ValueError("Richiesti almeno 1 blocco GPU e 64-256 token di generazione.")

    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(f"GPU non pronta: {status['errors']}; processi={status['foreign_processes']}")

    model = ModelManager().get("gemma-2-9b-it")
    if model is None:
        raise FileNotFoundError("Modello Gemma 2 9B non installato.")

    gpu = status["gpu"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} non corrisponde alla GPU scelta.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    if not llama_supports_gpu_offload():
        raise RuntimeError("llama-cpp-python non e' compilato con supporto offload CUDA.")

    from llama_cpp import Llama

    mib = 1024 * 1024
    n_ctx = 512
    report = {
        "phase": "model_mapping_benchmark",
        "model": model.name,
        "method": "llama-cpp-python",
        "version": version("llama-cpp-python"),
        "backend": "cuda",
        "gpu": gpu["name"],
        "gpu_layers": gpu_layers,
        "probe_n_ctx": n_ctx,
        "selected_n_ctx": None,
        "max_generated_tokens": max_tokens,
        "cpu_tensor_override": cpu_tensor,
        "measurements": {},
        "success": False,
        "error": None,
    }

    nvml.nvmlInit()
    stop = threading.Event()
    phase = ["load"]
    samples = {"load": [], "prefill": [], "decode": [], "close": []}
    sampling_errors = []
    llm = None
    sampler_thread = None

    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])

        def used_mib() -> float:
            return round(nvml.nvmlDeviceGetMemoryInfo(handle).used / mib, 1)

        def sample() -> None:
            while not stop.is_set():
                try:
                    samples[phase[0]].append(used_mib())
                except nvml.NVMLError as exc:
                    sampling_errors.append(str(exc))
                    break
                stop.wait(0.05)

        mem = report["measurements"]
        mem["vram_used_before_mib"] = used_mib()
        sampler_thread = threading.Thread(target=sample, daemon=True)
        sampler_thread.start()

        try:
            start = time.perf_counter()
            if cpu_tensor is None:
                llm = Llama(
                    model_path=str(model), n_gpu_layers=gpu_layers,
                    n_ctx=n_ctx, n_batch=128, n_ubatch=128,
                    offload_kqv=True, flash_attn=False, verbose=False,
                )
            else:
                from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

                llm, rules = LlamaCppPythonMappingAdapter().load(
                    model, n_gpu_layers=gpu_layers, n_ctx=n_ctx,
                    rules=[{
                        "pattern": "^" + re.escape(cpu_tensor) + "$",
                        "device": "cpu",
                    }],
                )
                report["mapping_rules"] = rules
            mem["load_seconds"] = round(time.perf_counter() - start, 3)
            mem["vram_used_loaded_mib"] = used_mib()

            # Prompt Gemma 2 in formato chat; piu' lungo della prova precedente
            # per ottenere abbastanza step di decode da confrontare.
            prompt = (
                "<start_of_turn>user\n"
                "Spiega dettagliatamente come funziona un motore elettrico, "
                "dalla conversione dell'energia alle forze magnetiche, "
                "con esempi pratici. Scrivi almeno 350 parole in italiano "
                "in un testo continuo, senza riassumere e senza elenchi."
                "<end_of_turn>\n<start_of_turn>model\n"
            )
            tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
            if len(tokens) + max_tokens >= n_ctx:
                raise ValueError("Il prompt e la generazione eccedono il contesto tecnico.")
            mem["prompt_tokens"] = len(tokens)

            phase[0] = "prefill"
            llm.reset()
            start = time.perf_counter()
            llm.eval(tokens)
            elapsed = time.perf_counter() - start
            mem["prefill_seconds"] = round(elapsed, 4)
            mem["prefill_tokens_per_second"] = round(len(tokens) / elapsed, 2)
            mem["vram_used_after_prefill_mib"] = used_mib()

            # generate([], reset=False) usa i logits prodotti da eval().
            # Il primo yield e' il campionamento iniziale, mentre gli
            # yield successivi includono il decode del token precedente.
            phase[0] = "decode"
            generator = llm.generate([], reset=False, temp=0.0, top_k=1)
            generated = 0
            ended_early = False
            eog = {llm.token_eos()}
            end_turn = llm.tokenize(b"<end_of_turn>", add_bos=False, special=True)
            if len(end_turn) == 1:
                eog.add(end_turn[0])

            started_first = time.perf_counter()
            first = None
            end = None
            for _ in range(max_tokens):
                token = next(generator)
                now = time.perf_counter()
                if first is None:
                    first = now
                end = now
                generated += 1
                if token in eog:
                    ended_early = True
                    break
            generator.close()

            mem["first_token_seconds_after_prefill"] = round(first - started_first, 4)
            mem["generated_tokens"] = generated
            mem["ended_at_eog"] = ended_early
            mem["decode_steps"] = max(0, generated - 1)
            mem["decode_seconds"] = round(end - first, 4)
            mem["decode_tokens_per_second"] = (
                round((generated - 1) / (end - first), 2)
                if generated > 1 and end > first else None
            )
            mem["vram_used_after_decode_mib"] = used_mib()

            # Un testo che finisce dopo pochi token non produce un dato di
            # throughput confrontabile. Non viene considerato benchmark valido.
            if generated < 48:
                raise RuntimeError(f"Campione decode insufficiente: {generated} token.")
            report["success"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            phase[0] = "close"
            if llm is not None:
                llm.close()
            stop.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=2)
            mem["vram_used_after_close_mib"] = used_mib()
            mem["vram_peak_sampled_mib"] = max(
                [mem["vram_used_before_mib"]]
                + [value for readings in samples.values() for value in readings]
            )
            mem["vram_peak_by_phase_mib"] = {
                name: max(readings) if readings else None
                for name, readings in samples.items()
            }
            mem["sample_count_by_phase"] = {
                name: len(readings) for name, readings in samples.items()
            }
            mem["vram_sampling_errors"] = sampling_errors
    finally:
        nvml.nvmlShutdown()

    after = GPUManager.calibration_status()
    report["foreign_processes_after"] = after["foreign_processes"]
    if not after["ready"] or sampling_errors:
        report["success"] = False
        report["error"] = report["error"] or (
            f"Misura non affidabile: stato GPU={after['errors']}; "
            f"processi esterni={after['foreign_processes']}; "
            f"errori campionamento={sampling_errors}"
        )

    report["measurement_note"] = (
        "Prefill misurato con eval; decode misurato sugli step successivi "
        "al primo token con generate. Picchi NVML campionati ogni 50 ms "
        "sulla GPU intera, non garantiti come massimi assoluti. "
        "Non e' un test di qualita' e non determina il contesto operativo."
    )
    return report


def benchmark(gpu_layers: int, max_tokens: int = 128, timeout: int = 180,
              cpu_tensor: str | None = None) -> dict:
    """Supervisiona la prova in subprocess; conserva diagnostica sui crash nativi."""
    if gpu_layers < 1 or not 64 <= max_tokens <= 256 or timeout < 1:
        raise ValueError("Richiesti gpu_layers>=1, 64<=tokens<=256 e timeout>0.")

    command = [
        sys.executable, "-m", "calibration.benchmark", "--worker",
        "--gpu-layers", str(gpu_layers), "--tokens", str(max_tokens),
    ]
    if cpu_tensor is not None:
        if not cpu_tensor or cpu_tensor == "token_embd.weight":
            raise ValueError("Tensore CPU non valido o condiviso con l'output.")
        command.extend(("--cpu-tensor", cpu_tensor))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False, "failure_type": "timeout",
            "gpu_layers": gpu_layers, "timeout_seconds": timeout,
            "error": f"Worker interrotto dopo {timeout} secondi",
        }

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        report = {
            "success": False,
            "gpu_layers": gpu_layers,
            "error": "Il worker non ha prodotto JSON valido.",
        }

    report["worker_returncode"] = result.returncode
    if result.returncode < 0:
        signum = -result.returncode
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = str(signum)
        report["failure_type"] = "native_signal"
        report["native_signal"] = signame
        report["success"] = False
    elif result.returncode and not report.get("failure_type"):
        report["failure_type"] = "worker_error"
        report["success"] = False

    if cpu_tensor is not None:
        from .granular_probe import _buffer_domain

        override_pattern = re.compile(
            r"tensor\s+" + re.escape(cpu_tensor)
            + r"\s+\([^\n]*?buffer type overridden to\s+([^\s]+)"
        )
        native_names = override_pattern.findall(result.stderr)
        domains = [_buffer_domain(name) for name in native_names]
        host_selected = bool(domains) and all(
            domain in ("host_cpu", "host_cuda_pinned") for domain in domains
        )
        output_gpu = "offloading output layer to GPU" in result.stderr
        report["mapping_verification"] = {
            "tensor": cpu_tensor,
            "native_buffer_names": native_names,
            "native_buffer_domains": domains,
            "tensor_host_selected": host_selected,
            "output_gpu_verified_by_log": output_gpu,
        }
        if not (host_selected and output_gpu):
            report["success"] = False
            report["error"] = report.get("error") or (
                "Mapping granulare non confermato dai log nativi."
            )

    # Mantieni le righe diagnostiche significative: i crash di CUDA
    # spesso includono path di build e warning del debugger poco utili.
    lines = [
        line.strip() for line in result.stderr.splitlines()
        if any(word in line.lower() for word in (
            "cuda error", "out of memory", "ggml_cuda", "cudamalloc",
            "failed to allocate", "assert", "abort", "sigabrt",
        ))
    ]
    report["cuda_diagnostics"] = [line[:400] for line in lines[-12:]]
    if result.returncode and not report["cuda_diagnostics"]:
        report["stderr_tail"] = result.stderr[-1500:]

    # Un generico errore CUDA o SIGABRT non e' prova certa di OOM.
    report["suspected_cuda_oom"] = any(
        term in result.stderr.lower()
        for term in ("out of memory", "cudamalloc failed", "cuda error 2")
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA isolato: prefill e decode separati"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cpu-tensor", default=None,
                        help="Tensore GGUF da forzare su host RAM (probe sperimentale)")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        # Evita file core enormi quando un'allocazione CUDA causa SIGABRT.
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            outcome = _worker(args.gpu_layers, args.tokens, args.cpu_tensor)
        except Exception as exc:
            outcome = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
    else:
        try:
            outcome = benchmark(args.gpu_layers, args.tokens, args.timeout,
                                cpu_tensor=args.cpu_tensor)
        except Exception as exc:
            print(f"Benchmark non avviato: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        if not outcome["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
