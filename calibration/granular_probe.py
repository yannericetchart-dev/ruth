"""Isolated diagnostic for native per-tensor placement with output on CUDA.

No production profile is changed. The native debug log is inspected to verify
both the output offload and the actual per-tensor buffer assignment.
"""

import argparse
import json
import os
import re
import resource
import signal
import subprocess
import sys
from pathlib import Path

from hardware import GPUManager
from inference.adapters.llama_cpp_python import LlamaCppPythonMappingAdapter
from model import ModelManager


def _worker(layers: int, target: str) -> dict:
    if layers < 1:
        raise ValueError("n_gpu_layers deve essere positivo.")
    status = GPUManager.calibration_status()
    if not status["ready"]:
        raise RuntimeError(f"GPU non pronta: {status['errors']}; {status['foreign_processes']}")
    path = ModelManager().get()
    if path is None:
        raise FileNotFoundError("Gemma 2 GGUF non installato.")

    gpu = status["gpu"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in (gpu["uuid"], str(gpu["index"])):
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES={visible!r} non compatibile.")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

    adapter = LlamaCppPythonMappingAdapter()
    pattern = "^" + re.escape(target) + "$"
    llm = None
    try:
        llm, rules = adapter.load(path, n_gpu_layers=layers, n_ctx=512, rules=[
            {"pattern": pattern, "device": "cpu"}
        ])
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Ciao Ruth, rispondi con una frase."}],
            max_tokens=32, temperature=0.0,
        )
        text = (result["choices"][0]["message"].get("content") or "").strip()
        if not text:
            raise RuntimeError("Generazione vuota.")
        return {
            "worker_ok": True,
            "model": path.name,
            "method": adapter.method,
            "backend": adapter.backend,
            "gpu_layers_requested": layers,
            "rules": rules,
            "probe_n_ctx": 512,
            "selected_n_ctx": None,
            "response_excerpt": text[:180],
        }
    finally:
        if llm is not None:
            llm.close()


def _buffer_domain(native_name: str) -> str:
    """Classify GGML buffer type names, not physical allocation guarantees.

    In llama.cpp's CUDA backend, CUDA_Host is host RAM (normally pinned),
    not GPU VRAM. Its allocator may fall back to ordinary CPU memory.
    """
    label = native_name.strip().upper()
    if label == "CUDA_HOST":
        return "host_cuda_pinned"
    if label == "CPU" or label.startswith("CPU_"):
        return "host_cpu"
    if label.startswith("CUDA") and "HOST" not in label:
        return "gpu_vram"
    return "unknown"


def probe(layers: int = 32, tensor: str | None = None, timeout: int = 180) -> dict:
    if layers < 1 or timeout <= 0:
        raise ValueError("Layer e timeout devono essere positivi.")

    # Select an actual tensor from an offloaded final transformer block,
    # rather than hardcode a particular block index into Ruth.
    layout = ModelManager().inspect(n_ctx=512)
    blocks = layout["metadata"].get("gemma2.block_count")
    if not isinstance(blocks, int) or layers > blocks + 1:
        raise ValueError(f"Layer richiesti {layers}; il modello dichiara {blocks} blocchi.")
    if tensor is None:
        tensor = f"blk.{blocks - 1}.ffn_down.weight"
    known = {item["name"] for item in layout["tensors"]}
    if tensor not in known or tensor == "token_embd.weight":
        raise ValueError(f"Tensore non utilizzabile per questo probe: {tensor}")

    command = [sys.executable, "-m", "calibration.granular_probe", "--worker",
               "--gpu-layers", str(layers), "--tensor", tensor]
    try:
        done = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"success": False, "failure_type": "timeout", "gpu_layers": layers}

    try:
        outcome = json.loads(done.stdout)
    except json.JSONDecodeError:
        outcome = {"worker_ok": False, "error": "Worker senza JSON valido."}

    logs = done.stderr
    output_on_gpu = "offloading output layer to GPU" in logs
    # In this version the native loader emits one debug line for every
    # override actually applied, including the chosen buffer type.
    pattern = re.compile(
        r"tensor\s+" + re.escape(tensor)
        + r"\s+\([^\n]*?buffer type overridden to\s+([^\s]+)"
    )
    applied = pattern.findall(logs)
    buffer_domains = [_buffer_domain(name) for name in applied]
    tensor_cpu = bool(buffer_domains) and all(
        domain in ("host_cpu", "host_cuda_pinned") for domain in buffer_domains
    )

    outcome.update({
        "phase": "granular_mapping_probe",
        "output_gpu_verified_by_log": output_on_gpu,
        "tensor_cpu_verified_by_log": tensor_cpu,
        "tensor": tensor,
        "native_buffer_names": applied,
        "native_buffer_domains": buffer_domains,
        "tensor_host_cuda_pinned_selected": "host_cuda_pinned" in buffer_domains,
        "worker_returncode": done.returncode,
        "success": done.returncode == 0 and outcome.get("worker_ok") is True
                   and output_on_gpu and tensor_cpu,
        "profile_generated": False,
        "log_note": (
            "Verifica del buffer selezionato tramite log DEBUG del loader nativo. "
            "CUDA_Host indica un buffer host CUDA (di norma RAM pinned), non VRAM; "
            "il backend puo' ripiegare su CPU ordinaria se l'allocazione pinned fallisce. "
            "Il log di offload output conferma l'assegnazione, non misura i byte allocati."
        ),
    })
    if done.returncode < 0:
        outcome["failure_type"] = "native_signal"
        outcome["native_signal"] = signal.Signals(-done.returncode).name
    elif done.returncode != 0:
        outcome["failure_type"] = "worker_error"
    if not outcome["success"]:
        diagnostics = [
            line.strip() for line in logs.splitlines()
            if any(word in line.lower() for word in (
                "tensor ", "output layer", "cuda error", "failed", "error", "override",
            ))
        ]
        outcome["diagnostics"] = diagnostics[-15:]
        if not diagnostics:
            outcome["stderr_tail"] = logs[-1000:]
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica mapping granulare GGUF")
    parser.add_argument("--gpu-layers", type=int, default=32)
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    try:
        if args.worker:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            result = _worker(args.gpu_layers, args.tensor)
        else:
            result = probe(args.gpu_layers, args.tensor, args.timeout)
    except Exception as exc:
        result = {"success": False, "worker_ok": False,
                  "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("success") and not result.get("worker_ok"):
        raise SystemExit(1)
    if not args.worker and not result.get("success"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
