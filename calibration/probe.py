"""Una singola prova di calibrazione CUDA, isolata dalla chat di Ruth."""

import argparse
import json
import os
import sys
import threading
import time
from importlib.metadata import version

from hardware import GPUManager
from model import ModelManager


def probe(gpu_layers: int, n_ctx: int, n_batch: int, n_ubatch: int, max_tokens: int) -> dict:
    if not (gpu_layers >= 1 and n_ctx >= 1 and n_batch >= 1 and 1 <= n_ubatch <= n_batch):
        raise ValueError("Parametri non validi: layer, contesto e batch devono essere positivi.")

    before = GPUManager.calibration_status()
    if not before["ready"]:
        raise RuntimeError(
            f"GPU non pronta: errori={before['errors']}; "
            f"processi esterni={before['foreign_processes']}"
        )

    path = ModelManager().get("gemma-2-9b-it")
    if path is None:
        raise FileNotFoundError("GGUF non installato: usa ModelManager.download()")

    gpu = before["gpu"]
    current_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current_devices and current_devices not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={current_devices!r} non corrisponde alla GPU selezionata."
        )
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    # Il binding deve poter usare CUDA davvero: non basta rilevare il driver.
    from llama_cpp.llama_cpp import llama_supports_gpu_offload
    if not llama_supports_gpu_offload():
        raise RuntimeError("llama-cpp-python nel venv non supporta CUDA.")

    import pynvml as nvml
    from llama_cpp import Llama

    report = {
        "model": path.name,
        "model_file_bytes": path.stat().st_size,
        "engine": "llama.cpp",
        "method": "llama-cpp-python",
        "package_version": version("llama-cpp-python"),
        "backend": "cuda",
        "gpu": {"name": gpu["name"], "uuid": gpu["uuid"], "vram_total_mib": gpu["vram_total_mb"]},
        "config": {
            "n_gpu_layers": gpu_layers,
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "offload_kqv": True,
            "flash_attn": False,
            "type_k": "runtime-default",
            "type_v": "runtime-default",
            "swa_full": None,
        },
        "measurements": {},
        "success": False,
        "error": None,
    }

    nvml.nvmlInit()
    stop = threading.Event()
    samples = []
    sampling_errors = []

    try:
        handle = nvml.nvmlDeviceGetHandleByUUID(gpu["uuid"])

        def used_mib() -> float:
            return round(nvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024), 1)

        def sample() -> None:
            while not stop.is_set():
                try:
                    samples.append(used_mib())
                except nvml.NVMLError as exc:
                    sampling_errors.append(str(exc))
                    break
                stop.wait(0.05)

        report["measurements"]["gpu_used_before_mib"] = used_mib()
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        llm = None

        try:
            start = time.perf_counter()
            llm = Llama(
                model_path=str(path),
                n_gpu_layers=gpu_layers,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_ubatch=n_ubatch,
                offload_kqv=True,
                flash_attn=False,
                verbose=False,
            )
            report["measurements"]["load_seconds"] = round(time.perf_counter() - start, 3)
            report["measurements"]["gpu_used_loaded_mib"] = used_mib()

            start = time.perf_counter()
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": "Ciao Ruth. Rispondi in una frase breve."}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            elapsed = time.perf_counter() - start
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens")

            report["measurements"].update({
                "generation_seconds": round(elapsed, 3),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": completion_tokens,
                "tokens_per_second_including_prefill": (
                    round(completion_tokens / elapsed, 2)
                    if completion_tokens is not None and elapsed > 0 else None
                ),
                "response_excerpt": (result["choices"][0]["message"].get("content") or "")[:160],
            })
            report["success"] = True
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if llm is not None:
                llm.close()
            stop.set()
            thread.join(timeout=2)
            report["measurements"]["gpu_used_after_close_mib"] = used_mib()
            report["measurements"]["gpu_used_peak_sampled_mib"] = max(
                samples + [report["measurements"]["gpu_used_before_mib"]]
            )
            report["measurements"]["samples"] = len(samples)
            report["measurements"]["sampling_errors"] = sampling_errors

    finally:
        nvml.nvmlShutdown()

    after = GPUManager.calibration_status()
    report["foreign_processes_after"] = after["foreign_processes"]
    if not after["ready"]:
        report["success"] = False
        report["error"] = report["error"] or (
            f"GPU non pronta dopo la prova: {after['errors']}; "
            f"processi esterni={after['foreign_processes']}"
        )
    if sampling_errors:
        report["success"] = False
        report["error"] = report["error"] or "Campionamento VRAM non affidabile."

    report["measurement_note"] = (
        "Picco VRAM campionato ogni 50 ms, non un massimo garantito. "
        "La prova misura memoria GPU globale e velocità complessiva inclusa la prefill; "
        "non stabilisce ancora la configurazione ottimale."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrazione preliminare: una prova CUDA")
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--tokens", type=int, default=64)
    args = parser.parse_args()
    try:
        result = probe(args.gpu_layers, args.ctx, args.batch, args.ubatch, args.tokens)
    except Exception as exc:
        print(f"Calibrazione non avviata: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
