"""Read-only GPU-guarded Gemma load/allocation diagnostics without inference.

No prompts, tokenizer, eval, decode, quantization auto-switch, profile publishing,
or CPU fallback. Requires a pinned verified local GGUF and explicit authorization.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .gpu_preflight import inspect_gpu_preflight
from .inference_probe import _valid_local_gguf, _select_unique_gpu, _parse_offload
from .load_only_worker import SCHEMA, PREFIX, SUPPORTED
from .install_plan import WHEEL_VERSION
from .model_catalog import select_model
from .native_failure_diagnostics import diagnose_cuda_failure
from .native_library import verify_native_library
from .venv_packages import inspect_python_environment

TIMEOUT = 360
WORKER_SCRIPT = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from ruth_runtime.load_only_worker import main;"
    "sys.exit(main(sys.argv[2:]))"
)
KV_SIZE = re.compile(r"\bkv_size\s*=\s*(\d+)\b")
BUFFER = re.compile(
    r"(?P<device>CUDA\d+|CPU(?:_[A-Za-z]+)?)\s+"
    r"(?:(?P<kind>model|KV|compute|output)\s+)?"
    r"buffer size\s*=\s*(?P<mib>\d+(?:\.\d+)?)\s+MiB", re.I,
)


def extract_native_buffers(log: str) -> list[dict]:
    """Native-reported allocation sizes, not inferred VRAM utilization."""
    observed = []
    for line in log.splitlines():
        match = BUFFER.search(line)
        if match:
            observed.append({
                "device": match.group("device"),
                "kind": (match.group("kind") or "unspecified").lower(),
                "mib": float(match.group("mib")),
            })
    return observed[:64]


def probe_model_load(
    *, project_root: Path = PROJECT_ROOT,
    quantization: str = "Q4_K_L",
    mode: str = "weights",
    n_ctx: int = 128,
    cuda_preinit: bool = False,
    allow_gpu_load: bool = False,
    environment: Any = None,
    run: Callable[..., Any] = subprocess.run,
    verify_native: Callable[..., Any] = verify_native_library,
    discover_gpu: Callable[..., Any] = _select_unique_gpu,
    gpu_preflight: Callable[..., Any] = inspect_gpu_preflight,
) -> dict:
    report = {
        "status": "gpu_load_not_authorized",
        "candidate": f"gemma-2-9B-it/gguf/{quantization}",
        "mode": mode, "cuda_preinit_requested": cuda_preinit,
        "weights_loaded": False,
        "context_initialized": False, "inference_executed": False,
        "generated_tokens": 0, "runtime_profile_published": False,
        "cpu_fallback_selected": False, "download_started": False,
    }
    def finish(status: str, **kw) -> dict:
        return {**report, "status": status, **kw}

    if not allow_gpu_load:
        return finish("gpu_load_not_authorized",
                      error="Explicit --allow-gpu-load is required")
    if (mode not in ("weights", "context") or quantization not in SUPPORTED
            or type(n_ctx) is not int or not 64 <= n_ctx <= 4096
            or type(cuda_preinit) is not bool):
        return finish("unsupported_load_diagnostic",
                      error="Only pinned Gemma Q4_K_L or Q4_K_M: weights/context; 64<=n_ctx<=4096")
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    if not (env.interpreter_in_project_venv
            and env.python_version.startswith("3.12.")
            and Path(env.project_root).resolve() == root.resolve()
            and Path(env.interpreter).resolve() == (root / ".venv/bin/python").resolve()):
        return finish("invalid_project_venv")
    try:
        choice = select_model("production", quantization=quantization, project_root=root)
        valid, _, reason = _valid_local_gguf(root, choice)
        if not valid:
            return finish("model_integrity_or_license_unverified", error=reason)
        native = verify_native(project_root=root, environment=env)
        if native.status != "native_library_dual_abi_verified":
            return finish("native_runtime_unverified", error=native.status)
        uuid, pci = discover_gpu(run=run)
        pre = gpu_preflight(uuid, run=run)
        if pre.status != "gpu_ready":
            return finish(pre.status, gpu_uuid=uuid, gpu_pci_bus=pci,
                          gpu_preflight=pre.to_dict(),
                          error=pre.error or "GPU occupied by host processes")
        worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
                      "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        worker_env.pop("LLAMA_CPP_LIB_PATH", None)
        argv = [str(root / ".venv/bin/python"), "-I", "-c", WORKER_SCRIPT,
                str(root), str(root), uuid, quantization, mode, str(n_ctx),
                "preinit" if cuda_preinit else "normal"]
        pre = gpu_preflight(uuid, run=run)
        if pre.status != "gpu_ready":
            return finish(pre.status, gpu_uuid=uuid, gpu_pci_bus=pci,
                          gpu_preflight=pre.to_dict(),
                          error=pre.error or "GPU changed before worker start")
        worker = run(argv, cwd=str(root), capture_output=True, text=True,
                     check=False, timeout=TIMEOUT, env=worker_env)
        native_buffers = extract_native_buffers(worker.stderr or "")
        native_kv_sizes = [int(x) for x in KV_SIZE.findall(worker.stderr or "")]
        lines = [s[len(PREFIX):] for s in (worker.stdout or "").splitlines()
                 if s.startswith(PREFIX)]
        if len(lines) != 1:
            diagnosis = diagnose_cuda_failure(worker.stderr or "",
                                              returncode=worker.returncode)
            return finish(
                "gpu_load_worker_failed_or_no_result",
                gpu_uuid=uuid, gpu_pci_bus=pci, gpu_preflight=pre.to_dict(),
                native_buffers=native_buffers, native_kv_sizes=native_kv_sizes,
                error="Worker did not emit a unique structured result",
                **diagnosis,
            )
        data = json.loads(lines[0])
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError("Unrecognized worker result schema")
        if (data.get("mode") != mode
            or data.get("quantization") != quantization
            or data.get("gpu_uuid") != uuid
            or data.get("inference_executed") is not False
            or data.get("generated_tokens") != 0
            or (data.get("cuda_preinit_requested") is not cuda_preinit)
            or (mode == "context" and data.get("n_ctx") != n_ctx)):
            raise ValueError("Unexpected worker identity or inference activity")
        weights = data.get("weights_loaded") is True
        context = data.get("context_initialized") is True
        layers, total = _parse_offload(worker.stderr or "")
        expected_success = weights and (mode == "weights" or context)
        success = worker.returncode == 0 and expected_success
        diagnosis = diagnose_cuda_failure(
            worker.stderr or "", worker_error=str(data.get("error") or ""),
            returncode=worker.returncode,
        )
        return finish(
            "gpu_load_only_completed" if success else "gpu_load_only_failed",
            gpu_uuid=uuid, gpu_pci_bus=pci, gpu_preflight=pre.to_dict(),
            model_sha256=getattr(choice, "expected_sha256", None),
            model_verified=True,
            source_repository=getattr(choice, "repo_id", None),
            source_revision=getattr(choice, "revision", None),
            native_binding_version=WHEEL_VERSION,
            weights_loaded=weights, context_initialized=context,
            n_ctx=data.get("n_ctx"), native_kv_sizes=native_kv_sizes,
            n_batch=data.get("n_batch"),
            n_ubatch=data.get("n_ubatch"),
            gpu_process_observed=data.get("gpu_process_observed") is True,
            gpu_memory_peak_mib_observed=data.get("gpu_memory_peak_mib_observed"),
            cuda_primary_context_preinitialized=(
                data.get("cuda_primary_context_preinitialized") is True),
            cuda_free_mib_after_backend_init=data.get("cuda_free_mib_after_backend_init"),
            cuda_free_mib_after_primary_ctx_init=data.get("cuda_free_mib_after_primary_ctx_init"),
            cuda_free_mib_after_model_load=data.get("cuda_free_mib_after_model_load"),
            cuda_free_mib_after_context_init=data.get("cuda_free_mib_after_context_init"),
            cuda_free_mib_after_failure=data.get("cuda_free_mib_after_failure"),
            native_buffers=native_buffers,
            offloaded_layers=layers, total_layers=total,
            full_gpu_offload_observed=(layers > 0 and layers == total),
            worker_exit_code=worker.returncode,
            native_failure_category=(None if success else diagnosis["native_failure_category"]),
            native_failure_signals=diagnosis["native_failure_signals"],
            native_log_excerpt=diagnosis["native_log_excerpt"],
            native_cuda_buffer_mib_reported=diagnosis["native_cuda_buffer_mib_reported"],
            error=None if success else str(data.get("error") or "No successful load")[:220],
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return finish("gpu_load_diagnostic_failed",
                      error=type(exc).__name__ + ": " + str(exc)[:260])
