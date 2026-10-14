"""Guarded Gemma 2 9B Q4_K_L CUDA minimum-context candidate probe.

No model download, license acceptance, quant switch or fallback. Must have an
existing SHA256-verified GGUF manifest. A successful probe never publishes a
production runtime profile: it is an input to Phase 09 context calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .inference_probe import _valid_local_gguf, _select_unique_gpu, _parse_offload
from .model_catalog import select_model
from .native_library import verify_native_library
from .venv_packages import inspect_python_environment
from .production_probe_worker import (
    SCHEMA, PREFIX, N_CTX, N_BATCH, N_UBATCH, MAX_TOKENS,
)

TIMEOUT = 360
COMMAND = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from ruth_runtime.production_probe_worker import main;"
    "sys.exit(main(sys.argv[2:]))"
)


@dataclass(frozen=True)
class ProductionCandidateProbe:
    status: str
    candidate: str
    model_verified: bool
    native_library_verified: bool
    model_loaded: bool
    n_ctx: int
    n_batch: int
    n_ubatch: int
    max_tokens: int
    generated_tokens: int
    offloaded_layers: int
    total_layers: int
    full_gpu_offload_verified: bool
    gpu_uuid: str | None
    gpu_pci_bus: str | None
    gpu_process_observed: bool
    gpu_memory_peak_mib_observed: int | None
    gpu_inference_verified: bool
    context_calibration_pending: bool
    runtime_profile_published: bool
    cpu_fallback_selected: bool
    download_started: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _empty(status: str, **updates) -> ProductionCandidateProbe:
    d = dict(
        status=status, candidate="gemma-2-9B-it/gguf/Q4_K_L",
        model_verified=False, native_library_verified=False,
        model_loaded=False, n_ctx=N_CTX, n_batch=N_BATCH,
        n_ubatch=N_UBATCH, max_tokens=MAX_TOKENS,
        generated_tokens=0, offloaded_layers=0, total_layers=0,
        full_gpu_offload_verified=False, gpu_uuid=None,
        gpu_pci_bus=None, gpu_process_observed=False,
        gpu_memory_peak_mib_observed=None, gpu_inference_verified=False,
        context_calibration_pending=True, runtime_profile_published=False,
        cpu_fallback_selected=False, download_started=False, error=None,
    )
    d.update(updates)
    return ProductionCandidateProbe(**d)


def probe_production_candidate(
    *, project_root: Path = PROJECT_ROOT,
    authorize_gpu_inference: bool = False,
    environment: Any = None,
    run: Callable[..., Any] = subprocess.run,
    verify_native: Callable[..., Any] = verify_native_library,
    discover_gpu: Callable[..., tuple[str, str]] = _select_unique_gpu,
) -> ProductionCandidateProbe:
    """Probe only after explicit GPU consent and verified local license manifest."""
    if not authorize_gpu_inference:
        return _empty("gpu_inference_not_authorized",
                      error="Explicit --allow-gpu-inference required")
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    if not (env.interpreter_in_project_venv
            and env.python_version.startswith("3.12.")
            and Path(env.project_root).resolve() == root.resolve()
            and Path(env.interpreter).resolve() ==
                (root / ".venv/bin/python").resolve()):
        return _empty("invalid_project_venv")
    try:
        selected = select_model("production", project_root=root)
        if (selected.family, selected.model, selected.version,
                selected.quantization) != (
                    "gemma", "gemma-2", "gemma-2-9B-it", "Q4_K_L"):
            return _empty("unexpected_production_candidate")
        verified, model, reason = _valid_local_gguf(root, selected)
        if not verified:
            return _empty("model_integrity_or_license_unverified", error=reason)
        native = verify_native(project_root=root, environment=env)
        if native.status != "native_library_dual_abi_verified":
            return _empty("native_runtime_unverified", model_verified=True,
                          error=native.status)
        uuid, pci = discover_gpu(run=run)
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
                     "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        child_env.pop("LLAMA_CPP_LIB_PATH", None)
        cmd = [str(root / ".venv/bin/python"), "-I", "-c", COMMAND,
               str(root), str(root), uuid]
        proc = run(
            cmd, cwd=str(root), capture_output=True, text=True,
            check=False, timeout=TIMEOUT, env=child_env,
        )
        results = [line[len(PREFIX):] for line in (proc.stdout or "").splitlines()
                   if line.startswith(PREFIX)]
        if len(results) != 1:
            return _empty("worker_failed_or_no_result", model_verified=True,
                          native_library_verified=True, gpu_uuid=uuid,
                          gpu_pci_bus=pci,
                          error="No unique structured result; native OOM/crash possible")
        result = json.loads(results[0])
        if not isinstance(result, dict) or result.get("schema") != SCHEMA:
            raise ValueError("Unrecognized worker result")
        if proc.returncode:
            return _empty("production_min_context_worker_failed",
                          model_verified=True, native_library_verified=True,
                          gpu_uuid=uuid, gpu_pci_bus=pci,
                          error=str(result.get("error") or
                                    "Worker ended without a successful generation")[:220])
        if result.get("purpose") != "production" or \
                result.get("gpu_uuid") != uuid or \
                type(result.get("worker_pid")) is not int or \
                result["worker_pid"] < 1 or \
                result.get("n_ctx") != N_CTX or \
                result.get("n_batch") != N_BATCH or \
                result.get("n_ubatch") != N_UBATCH or \
                result.get("max_tokens") != MAX_TOKENS or \
                result.get("model_cache_enabled") is not False or \
                result.get("cpu_fallback_selected") is not False:
            raise ValueError("Worker identity/configuration mismatch")
        tokens = result.get("completion_tokens")
        mem = result.get("gpu_memory_peak_mib_observed")
        duration = result.get("worker_duration_seconds")
        if type(tokens) is not int or tokens < 0 or tokens > MAX_TOKENS or \
                type(mem) is not int or mem <= 0 or \
                type(duration) not in (int, float) or \
                not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid worker tokens/memory/timing")
        layers, total = _parse_offload(proc.stderr or "")
        full = layers > 0 and layers == total
        observed = result.get("gpu_process_observed") is True
        loaded = result.get("model_loaded") is True
        passed = full and observed and loaded and tokens > 0
        return _empty(
            "gemma_min_context_gpu_verified" if passed
            else "gemma_min_context_inconclusive",
            model_verified=True, native_library_verified=True,
            gpu_uuid=uuid, gpu_pci_bus=pci,
            model_loaded=loaded, generated_tokens=tokens,
            offloaded_layers=layers, total_layers=total,
            full_gpu_offload_verified=full,
            gpu_process_observed=observed,
            gpu_memory_peak_mib_observed=mem,
            gpu_inference_verified=passed,
            error=None if passed else
            "Incomplete full CUDA offload, GPU PID or token evidence",
        )
    except (OSError, RuntimeError, ValueError, TypeError,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return _empty("gemma_min_context_probe_failed",
                      error=type(exc).__name__ + ": " + str(exc)[:250])
