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


def probe_prompt(repeat: int = 1) -> str:
    """One Gemma-2 chat turn with a configurable deterministic prompt length."""
    if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= 48:
        raise ValueError("prompt_repeat deve essere compreso fra 1 e 48.")
    body = (
        "Spiega dettagliatamente come funziona un motore elettrico, "
        "dalla conversione dell'energia alle forze magnetiche, "
        "con esempi pratici. Scrivi almeno 350 parole in italiano "
        "in un testo continuo, senza riassumere e senza elenchi."
    )
    # Keep the previous exact prompt for repeat=1, so historical
    # 512-context measurements remain comparable.
    return (
        "<start_of_turn>user\n"
        + " ".join([body] * repeat)
        + "<end_of_turn>\n<start_of_turn>model\n"
    )


def validate_probe(n_ctx: int, prompt_repeat: int, max_tokens: int) -> None:
    if (not isinstance(n_ctx, int) or isinstance(n_ctx, bool)
        or n_ctx < 512 or n_ctx > 8192):
        raise ValueError("probe_n_ctx diagnostico deve essere 512..8192.")
    probe_prompt(prompt_repeat)
    if not isinstance(max_tokens, int) or not 64 <= max_tokens <= 256:
        raise ValueError("tokens deve essere 64..256.")


def _worker(gpu_layers: int, max_tokens: int, cpu_tensor: str | None = None,
            candidate_id: str | None = None, native_memory_audit: bool = False,
            combination_ids: list[str] | None = None,
            collect_gpu_telemetry: bool = False,
            probe_n_ctx: int = 512, prompt_repeat: int = 1) -> dict:
    import pynvml as nvml

    validate_probe(probe_n_ctx, prompt_repeat, max_tokens)
    if gpu_layers < 1 or max_tokens < 64 or max_tokens > 256:
        raise ValueError("Richiesti almeno 1 blocco GPU e 64-256 token di generazione.")
    if sum(x is not None for x in (cpu_tensor, candidate_id, combination_ids)) > 1:
        raise ValueError("Usa una sola modalita': cpu_tensor, candidate_id o combination_ids.")

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
    n_ctx = probe_n_ctx
    report = {
        "phase": "model_mapping_benchmark",
        "model": model.name,
        "method": "llama-cpp-python",
        "version": version("llama-cpp-python"),
        "backend": "cuda",
        "gpu": gpu["name"],
        "gpu_layers": gpu_layers,
        "probe_n_ctx": n_ctx,
        "prompt_repeat": prompt_repeat,
        "selected_n_ctx": None,
        "max_generated_tokens": max_tokens,
        "cpu_tensor_override": cpu_tensor,
        "mapping_candidate_id": candidate_id,
        "mapping_combination_ids": combination_ids,
        "native_memory_audit_requested": native_memory_audit,
        "gpu_telemetry_requested": collect_gpu_telemetry,
        "measurements": {},
        "success": False,
        "error": None,
    }

    nvml.nvmlInit()
    stop = threading.Event()
    phase = ["load"]
    samples = {"load": [], "prefill": [], "decode": [], "close": []}
    gpu_samples = {"load": [], "prefill": [], "decode": [], "close": []}
    sampling_errors = []
    llm = None
    sampler_thread = None

    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])

        def used_mib() -> float:
            return round(nvml.nvmlDeviceGetMemoryInfo(handle).used / mib, 1)

        def sample() -> None:
            tick = 0
            while not stop.is_set():
                try:
                    now_phase = phase[0]
                    samples[now_phase].append(used_mib())
                    if collect_gpu_telemetry and tick % 4 == 0:
                        from .gpu_telemetry import gpu_snapshot
                        gpu_samples[now_phase].append(gpu_snapshot(nvml, handle))
                    tick += 1
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
            if cpu_tensor is None and candidate_id is None and combination_ids is None:
                llm = Llama(
                    model_path=str(model), n_gpu_layers=gpu_layers,
                    n_ctx=n_ctx, n_batch=128, n_ubatch=128,
                    offload_kqv=True, flash_attn=False,
                    verbose=native_memory_audit,
                )
            else:
                from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter

                if candidate_id is not None or combination_ids is not None:
                    from .mapping_candidates import propose_mapping_candidates
                    from .mapping_composition import compose_candidates

                    layout = ModelManager().inspect("gemma-2-9b-it", n_ctx=n_ctx)
                    proposals = propose_mapping_candidates(layout, gpu_layers)
                    if combination_ids is not None:
                        candidate = compose_candidates(proposals, combination_ids)
                    else:
                        candidate = next(
                            (item for item in proposals["candidates"]
                             if item["id"] == candidate_id), None
                        )
                    if candidate is None or not candidate["rules"]:
                        raise ValueError("Candidato mapping non valido.")
                    requested_rules = candidate["rules"]
                else:
                    requested_rules = [{
                        "pattern": "^" + re.escape(cpu_tensor) + "$",
                        "device": "cpu",
                    }]
                llm, rules = LlamaCppPythonMappingAdapter().load(
                    model, n_gpu_layers=gpu_layers, n_ctx=n_ctx,
                    rules=requested_rules,
                )
                report["mapping_rules"] = rules
            mem["load_seconds"] = round(time.perf_counter() - start, 3)
            mem["vram_used_loaded_mib"] = used_mib()

            # Prompt Gemma 2 in formato chat; piu' lungo della prova precedente
            # per ottenere abbastanza step di decode da confrontare.
            prompt = probe_prompt(prompt_repeat)
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
            generation_timestamps = []
            for _ in range(max_tokens):
                token = next(generator)
                now = time.perf_counter()
                generation_timestamps.append(now)
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
            if generated >= 5:
                from .decode_windows import summarize_decode_windows
                mem["decode_windows"] = summarize_decode_windows(
                    generation_timestamps, len(tokens), n_ctx
                )
            else:
                mem["decode_windows"] = None

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
            if collect_gpu_telemetry:
                from .gpu_telemetry import summarize_gpu_telemetry
                report["gpu_telemetry_by_phase"] = summarize_gpu_telemetry(gpu_samples)
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
        "I quartili decode indicano gli intervalli di token generati, "
        "non la memoria KV assegnata. "
        "Non e' un test di qualita' e non determina il contesto operativo."
    )
    return report


def benchmark(gpu_layers: int, max_tokens: int = 128, timeout: int = 180,
              cpu_tensor: str | None = None, candidate_id: str | None = None,
              native_memory_audit: bool = False,
              combination_ids: list[str] | None = None,
              collect_gpu_telemetry: bool = False,
              probe_n_ctx: int = 512, prompt_repeat: int = 1) -> dict:
    """Supervisiona la prova in subprocess; conserva diagnostica sui crash nativi."""
    validate_probe(probe_n_ctx, prompt_repeat, max_tokens)
    if gpu_layers < 1 or timeout < 1:
        raise ValueError("Richiesti gpu_layers>=1 e timeout>0.")
    if sum(x is not None for x in (cpu_tensor, candidate_id, combination_ids)) > 1:
        raise ValueError("Non combinare cpu_tensor, candidate_id e combination_ids.")

    candidate = None
    if candidate_id is not None or combination_ids is not None:
        from .mapping_candidates import propose_mapping_candidates
        from .mapping_composition import compose_candidates

        proposals = propose_mapping_candidates(
            ModelManager().inspect("gemma-2-9b-it", n_ctx=probe_n_ctx), gpu_layers
        )
        if combination_ids is not None:
            candidate = compose_candidates(proposals, combination_ids)
        else:
            candidate = next(
                (item for item in proposals["candidates"]
                 if item["id"] == candidate_id), None
            )
        if candidate is None or not candidate["rules"]:
            raise ValueError(f"Candidato mapping sconosciuto: {candidate_id}")

    command = [
        sys.executable, "-m", "calibration.benchmark", "--worker",
        "--gpu-layers", str(gpu_layers), "--tokens", str(max_tokens),
        "--probe-n-ctx", str(probe_n_ctx),
        "--prompt-repeat", str(prompt_repeat),
    ]
    if cpu_tensor is not None:
        if not cpu_tensor or cpu_tensor == "token_embd.weight":
            raise ValueError("Tensore CPU non valido o condiviso con l'output.")
        command.extend(("--cpu-tensor", cpu_tensor))
    if candidate_id is not None:
        command.extend(("--candidate-id", candidate_id))
    if combination_ids is not None:
        for id_ in combination_ids:
            command.extend(("--combine", id_))
    if native_memory_audit:
        command.append("--native-memory-audit")
    if collect_gpu_telemetry:
        command.append("--gpu-telemetry")
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

    if candidate is not None:
        from .mapping_native_validation import verify_override_logs

        verified = verify_override_logs(candidate, result.stderr)
        report["mapping_verification"] = verified
        if not verified["verified"]:
            report["success"] = False
            report["error"] = report.get("error") or (
                "Il mapping granulare non corrisponde ai log nativi."
            )

    if native_memory_audit:
        from .native_memory import parse_native_memory

        report["native_memory_breakdown"] = parse_native_memory(result.stderr)
        report["native_memory_breakdown"]["logs_enabled_for_baseline"] = True

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

    # La diagnostica e' importante anche quando il worker Python termina
    # con returncode=0: llama_cpp puo' segnalare il fallimento della
    # creazione di llama_context come ValueError senza segnale nativo.
    from .worker_failure import analyze_worker_failure

    report.update(analyze_worker_failure(
        result.stderr, report.get("error"),
        result.returncode, report.get("success") is True,
    ))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA isolato: prefill e decode separati"
    )
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--probe-n-ctx", type=int, default=512,
                        help="Contesto della prova, NON valore operativo")
    parser.add_argument("--prompt-repeat", type=int, default=1,
                        help="Lunghezza del testo input diagnostico (1..48)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--cpu-tensor", default=None,
                        help="Tensore GGUF da forzare su host RAM (probe sperimentale)")
    parser.add_argument("--candidate-id", default=None,
                        help="ID candidato del generatore GGUF (diagnostico)")
    parser.add_argument("--combine", action="append", default=None,
                        help="Ripeti per combinare candidati senza override sovrapposti")
    parser.add_argument("--native-memory-audit", action="store_true",
                        help="Include buffer nativi model/KV/compute/output nel report")
    parser.add_argument("--gpu-telemetry", action="store_true",
                        help="Campiona temperatura, clock SM e potenza NVML per fase")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        # Evita file core enormi quando un'allocazione CUDA causa SIGABRT.
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            outcome = _worker(args.gpu_layers, args.tokens, args.cpu_tensor,
                              args.candidate_id, args.native_memory_audit,
                              args.combine, args.gpu_telemetry,
                              args.probe_n_ctx, args.prompt_repeat)
        except Exception as exc:
            outcome = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
    else:
        try:
            outcome = benchmark(args.gpu_layers, args.tokens, args.timeout,
                                cpu_tensor=args.cpu_tensor,
                                candidate_id=args.candidate_id,
                                native_memory_audit=args.native_memory_audit,
                                combination_ids=args.combine,
                                collect_gpu_telemetry=args.gpu_telemetry,
                                probe_n_ctx=args.probe_n_ctx,
                                prompt_repeat=args.prompt_repeat)
        except Exception as exc:
            print(f"Benchmark non avviato: {exc}", file=sys.stderr)
            raise SystemExit(1) from None
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        if not outcome["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
