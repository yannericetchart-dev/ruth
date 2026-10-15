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
KV_FORMATS = {"F16": "GGML_TYPE_F16", "Q8_0": "GGML_TYPE_Q8_0",
              "Q4_0": "GGML_TYPE_Q4_0"}
FLASH_FORMATS = {"off": "LLAMA_FLASH_ATTN_TYPE_DISABLED",
                 "auto": "LLAMA_FLASH_ATTN_TYPE_AUTO",
                 "on": "LLAMA_FLASH_ATTN_TYPE_ENABLED"}


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



def initialize_cuda_primary_context() -> tuple[ctypes.CDLL, int]:
    """Diagnostic-only: activate device 0 primary context before GGUF loading.

    CUDA_VISIBLE_DEVICES is already restricted by the guarded parent worker.
    This intentionally changes initialization order; compare with a normal
    unmodified weights-only load in a different worker process.
    """
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuInit.restype = ctypes.c_int
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGet.restype = ctypes.c_int
    cuda.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
    ]
    cuda.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
    cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    cuda.cuCtxSetCurrent.restype = ctypes.c_int
    cuda.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
    cuda.cuDevicePrimaryCtxRelease.restype = ctypes.c_int

    def require(name: str, status: int):
        if status != 0:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    require("cuInit", cuda.cuInit(0))
    device = ctypes.c_int()
    require("cuDeviceGet", cuda.cuDeviceGet(ctypes.byref(device), 0))
    context = ctypes.c_void_p()
    require("cuDevicePrimaryCtxRetain",
            cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device.value))
    try:
        require("cuCtxSetCurrent", cuda.cuCtxSetCurrent(context))
    except Exception:
        cuda.cuDevicePrimaryCtxRelease(device.value)
        raise
    return cuda, device.value

def run_load_only(root: Path, gpu_uuid: str, quantization: str, mode: str,
                  n_ctx: int = N_CTX, cuda_preinit: bool = False) -> dict:
    if (mode not in ("weights", "context") or quantization not in SUPPORTED
            or type(n_ctx) is not int or not 64 <= n_ctx <= 4096
            or type(cuda_preinit) is not bool):
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

    # Parent-only, strict serialized configuration. The worker never treats
    # arbitrary environment entries as native option names.
    raw_options = os.environ.get("RUTH_LLAMA_KV_OPTIONS")
    settings = (json.loads(raw_options) if raw_options else {
        "n_batch": N_BATCH, "n_ubatch": N_UBATCH,
        "swa_full": True, "flash_attention": "auto",
        "type_k": "F16", "type_v": "F16",
    })
    if (not isinstance(settings, dict) or
        set(settings) != {"n_batch", "n_ubatch", "swa_full",
                          "flash_attention", "type_k", "type_v"}
        or type(settings["n_batch"]) is not int
        or type(settings["n_ubatch"]) is not int
        or not 1 <= settings["n_ubatch"] <= settings["n_batch"] <= n_ctx
        or type(settings["swa_full"]) is not bool
        or settings["flash_attention"] not in FLASH_FORMATS
        or settings["type_k"] not in KV_FORMATS
        or settings["type_v"] not in KV_FORMATS
        or (settings["flash_attention"] == "off"
            and settings["type_v"] != "F16")):
        raise ValueError("Unapproved native KV cache parameters")
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
        "n_batch": settings["n_batch"] if mode == "context" else None,
        "n_ubatch": settings["n_ubatch"] if mode == "context" else None,
        "context_options_requested": ({
            "n_ctx": n_ctx, **settings,
            "n_seq_max": 1, "kv_unified": False,
            "offload_kqv": True,
        } if mode == "context" else None),
        "cuda_preinit_requested": cuda_preinit,
        "cuda_primary_context_preinitialized": False,
        "cuda_free_mib_after_backend_init": None,
        "cuda_free_mib_after_primary_ctx_init": None,
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
    primary_context_driver = None
    primary_context_device = None
    start = monotonic()
    watcher.start()
    try:
        api.llama_backend_init()
        backend_initialized = True
        report["cuda_free_mib_after_backend_init"] = current_cuda_free_mib()
        if cuda_preinit:
            primary_context_driver, primary_context_device = initialize_cuda_primary_context()
            report["cuda_primary_context_preinitialized"] = True
            report["cuda_free_mib_after_primary_ctx_init"] = current_cuda_free_mib()
            if report["cuda_free_mib_after_primary_ctx_init"] is None:
                raise RuntimeError("Primary CUDA context initialized but cuMemGetInfo unavailable")
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
            options.n_batch = settings["n_batch"]
            options.n_ubatch = settings["n_ubatch"]
            options.offload_kqv = True
            options.swa_full = settings["swa_full"]
            options.n_seq_max = 1
            options.kv_unified = False
            options.flash_attn_type = getattr(
                api, FLASH_FORMATS[settings["flash_attention"]])
            options.type_k = getattr(api, KV_FORMATS[settings["type_k"]])
            options.type_v = getattr(api, KV_FORMATS[settings["type_v"]])
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
            if primary_context_driver is not None:
                # Release only our additional explicit primary-context retain;
                # never reset other CUDA state.
                primary_context_driver.cuCtxSetCurrent(ctypes.c_void_p())
                primary_context_driver.cuDevicePrimaryCtxRelease(
                    primary_context_device)

            stop.set()
            watcher.join(timeout=5)
            report["gpu_process_observed"] = bool(samples)
            report["gpu_memory_peak_mib_observed"] = max(samples) if samples else None
            report["duration_seconds"] = round(monotonic() - start, 3)
    return report


def main(args: list[str] | None = None) -> int:
    argv = sys.argv[1:] if args is None else args
    if len(argv) != 6 or argv[5] not in ("normal", "preinit"):
        raise SystemExit("Expected root, GPU UUID, quantization, mode, n_ctx, normal|preinit")
    try:
        result = run_load_only(Path(argv[0]), argv[1], argv[2], argv[3],
                               int(argv[4]), argv[5] == "preinit")
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
