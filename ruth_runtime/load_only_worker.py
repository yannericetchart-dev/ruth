"""Isolated pinned CUDA load diagnostic: NO tokenization, eval or generation.

Modes:
- weights: llama_model_load_from_file only (no context/KV)
- context: weights then llama_init_from_model (KV/compute allocation)
Runtime buffer logs are captured by the parent from native stderr.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
from time import monotonic

from .inference_worker import measure_worker_gpu
from .model_catalog import select_model
from .model_download import _local_path

SCHEMA = "ruth-cuda-load-only-v1"
PREFIX = "RUTH_CUDA_LOAD_ONLY_RESULT="
N_CTX, N_BATCH, N_UBATCH = 128, 64, 64
SUPPORTED = frozenset({"Q4_K_L", "Q4_K_M"})


def current_cuda_free_mib() -> float | None:
    """Query the existing CUDA context without creating a second context."""
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        current = ctypes.c_void_p()
        cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cuda.cuCtxGetCurrent.restype = ctypes.c_int
        if cuda.cuCtxGetCurrent(ctypes.byref(current)) != 0 or not current.value:
            return None
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        cuda.cuMemGetInfo_v2.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ]
        cuda.cuMemGetInfo_v2.restype = ctypes.c_int
        if cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != 0:
            return None
        return round(free.value / 1048576, 2)
    except (OSError, AttributeError, ValueError):
        return None


def run_load_only(root: Path, gpu_uuid: str, quantization: str, mode: str, n_ctx: int = N_CTX) -> dict:
    if (mode not in ("weights", "context") or quantization not in SUPPORTED
            or type(n_ctx) is not int or not 64 <= n_ctx <= 4096):
        raise ValueError("Unapproved pinned quantization or load-only mode")
    root = root.resolve(strict=True)
    choice = select_model("production", quantization=quantization, project_root=root)
    if (choice.family, choice.model, choice.version, choice.quantization) != (
        "gemma", "gemma-2", "gemma-2-9B-it", quantization,
    ):
        raise RuntimeError("Unexpected model for Gemma CUDA load diagnostic")
    file = _local_path(root, choice)
    if file.is_symlink() or not file.is_file() or file.stat().st_size != choice.expected_size_bytes:
        raise RuntimeError("Local pinned GGUF not present or changed")
    with file.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise RuntimeError("Invalid local GGUF")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("Worker GPU UUID binding mismatch")

    from llama_cpp import llama_cpp as api
    from llama_cpp import __version__ as runtime_version
    from .install_plan import WHEEL_VERSION
    if runtime_version != WHEEL_VERSION:
        raise RuntimeError("Unexpected llama-cpp-python version")
    if not api.llama_supports_gpu_offload():
        raise RuntimeError("CUDA offload unavailable")

    samples: list[int] = []
    stop = Event()
    watcher = Thread(target=measure_worker_gpu,
                     args=(os.getpid(), gpu_uuid, stop, samples), daemon=True)
    report = {
        "schema": SCHEMA, "mode": mode, "quantization": quantization,
        "gpu_uuid": gpu_uuid, "worker_pid": os.getpid(),
        "weights_loaded": False, "context_initialized": False,
        "inference_executed": False, "generated_tokens": 0,
        "n_ctx": n_ctx if mode == "context" else None,
        "n_batch": N_BATCH if mode == "context" else None,
        "n_ubatch": N_UBATCH if mode == "context" else None,
        "cuda_free_mib_after_backend_init": None,
        "cuda_free_mib_after_model_load": None,
        "cuda_free_mib_after_context_init": None,
        "cuda_free_mib_after_failure": None,
        "gpu_process_observed": False,
        "gpu_memory_peak_mib_observed": None,
        "error": None,
    }
    model = None
    context = None
    backend_initialized = False
    start = monotonic()
    watcher.start()
    try:
        api.llama_backend_init()
        backend_initialized = True
        report["cuda_free_mib_after_backend_init"] = current_cuda_free_mib()
        params = api.llama_model_default_params()
        params.n_gpu_layers = 2**31 - 1
        params.main_gpu = 0
        params.use_mmap = True
        model = api.llama_model_load_from_file(os.fsencode(file), params)
        if not model:
            raise RuntimeError("llama_model_load_from_file returned null")
        report["weights_loaded"] = True
        report["cuda_free_mib_after_model_load"] = current_cuda_free_mib()
        if mode == "context":
            options = api.llama_context_default_params()
            options.n_ctx = n_ctx
            options.n_batch = N_BATCH
            options.n_ubatch = N_UBATCH
            options.offload_kqv = True
            context = api.llama_init_from_model(model, options)
            if not context:
                raise RuntimeError("llama_init_from_model returned null")
            report["context_initialized"] = True
            report["cuda_free_mib_after_context_init"] = current_cuda_free_mib()
    except (OSError, RuntimeError, ValueError, MemoryError, ImportError) as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)[:220]
        report["cuda_free_mib_after_failure"] = current_cuda_free_mib()
    finally:
        try:
            if context:
                api.llama_free(context)
            if model:
                api.llama_model_free(model)
            if backend_initialized:
                api.llama_backend_free()
        finally:
            stop.set()
            watcher.join(timeout=5)
            report["gpu_process_observed"] = bool(samples)
            report["gpu_memory_peak_mib_observed"] = max(samples) if samples else None
            report["duration_seconds"] = round(monotonic() - start, 3)
    return report


def main(args: list[str] | None = None) -> int:
    argv = sys.argv[1:] if args is None else args
    if len(argv) != 5:
        raise SystemExit("Expected project root, GPU UUID, quantization, mode, n_ctx")
    try:
        result = run_load_only(Path(argv[0]), argv[1], argv[2], argv[3], int(argv[4]))
        print(PREFIX + json.dumps(result, sort_keys=True))
        return 0 if result["weights_loaded"] and (
            argv[3] == "weights" or result["context_initialized"]
        ) else 2
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(PREFIX + json.dumps({
            "schema": SCHEMA, "mode": argv[3], "quantization": argv[2],
            "gpu_uuid": argv[1], "weights_loaded": False,
            "context_initialized": False, "inference_executed": False,
            "generated_tokens": 0,
            "error": type(exc).__name__ + ": " + str(exc)[:220],
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
