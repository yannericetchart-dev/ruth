"""Fail-closed parent for Ruth's first real CUDA GGUF inference smoke.

All native model operations occur in a bounded child process of Ruth's .venv.
Success requires real completion tokens, an offloaded layer count in llama.cpp
diagnostics, AND nvidia-smi observing worker PID/selected GPU UUID.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Callable, Any

from .dependencies import PROJECT_ROOT
from .model_catalog import ModelChoice, select_model
from .model_download import _local_path, _manifest
from .native_library import verify_native_library
from .venv_packages import inspect_python_environment

OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", re.I)
WORKER_PREFIX = "RUTH_GPU_WORKER_RESULT="
TIMEOUT_SECONDS = 240
WORKER_SCRIPT = (
    "import sys;"
    "sys.path.insert(0,sys.argv[1]);"
    "from ruth_runtime.inference_worker import main;"
    "sys.exit(main(sys.argv[2:]))"
)


@dataclass(frozen=True)
class GpuInferenceReport:
    status: str
    model_file: str | None
    model_verified: bool
    native_library_verified: bool
    gpu_uuid: str | None
    gpu_pci_bus: str | None
    model_loaded: bool
    completion_tokens: int
    offloaded_layers: int
    total_layers: int
    gpu_process_observed: bool
    gpu_memory_peak_mib_observed: int | None
    gpu_offload_executed: bool
    inference_executed: bool
    gpu_inference_verified: bool
    duration_seconds: float | None
    cpu_fallback_selected: bool
    diskcache_todo_open: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _report(status: str, *, model: Path | None = None, verified: bool = False,
            native: bool = False, uuid: str | None = None, pci: str | None = None,
            loaded: bool = False, tokens: int = 0, offloaded: int = 0,
            total: int = 0, process: bool = False, memory: int | None = None,
            duration: float | None = None, error: str | None = None) -> GpuInferenceReport:
    # GPU execution is never inferred merely from a supported CUDA symbol.
    has_gpu_evidence = bool(offloaded > 0 and total >= offloaded and process
                            and memory is not None and memory > 0)
    actual_inference = bool(loaded and tokens > 0)
    verified_all = bool(status == "gpu_gguf_inference_verified" and
                        verified and native and uuid and pci and
                        has_gpu_evidence and actual_inference)
    return GpuInferenceReport(
        status=status, model_file=str(model) if model else None,
        model_verified=verified, native_library_verified=native,
        gpu_uuid=uuid, gpu_pci_bus=pci, model_loaded=loaded,
        completion_tokens=tokens, offloaded_layers=offloaded,
        total_layers=total, gpu_process_observed=process,
        gpu_memory_peak_mib_observed=memory,
        gpu_offload_executed=has_gpu_evidence,
        inference_executed=actual_inference,
        gpu_inference_verified=verified_all,
        duration_seconds=duration, cpu_fallback_selected=False,
        diskcache_todo_open=True, error=error,
    )


def _valid_local_gguf(root: Path, choice: ModelChoice) -> tuple[bool, Path, str | None]:
    """Verify full pinned model hash+byte size and completed sidecar manifest."""
    target = _local_path(root, choice)
    sidecar = target.with_name(target.name + ".manifest.json")
    if not choice.revision or not choice.expected_size_bytes:
        return False, target, "Model has no pinned size/revision"
    if any(p.is_symlink() for p in (
        root / "models", *list(target.parents)[:-1], target, sidecar,
    )):
        return False, target, "Model contains symlinked path"
    try:
        if not target.is_file() or not sidecar.is_file():
            return False, target, "Complete GGUF and verified manifest required"
        if target.stat().st_size != choice.expected_size_bytes:
            return False, target, "Model size differs from immutable catalog"
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(raw.get(k) != v for k, v in _manifest(choice).items()) or \
                raw.get("verified") is not True or \
                raw.get("license_acknowledged_for_this_download") is not True:
            return False, target, "Local model manifest not verified/pinned"
        digest = hashlib.sha256()
        with target.open("rb") as fh:
            if fh.read(4) != b"GGUF":
                return False, target, "Not a GGUF model"
            fh.seek(0)
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != choice.expected_sha256:
            return False, target, "Local GGUF SHA256 changed"
        return True, target, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return False, target, type(exc).__name__ + ": " + str(exc)[:150]


def _select_unique_gpu(*, run: Callable[..., Any] = subprocess.run) -> tuple[str, str]:
    result = run(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    if result.returncode:
        raise RuntimeError("nvidia-smi GPU inventory failed")
    records = []
    for line in result.stdout.splitlines():
        parts = [s.strip() for s in line.split(",")]
        if len(parts) != 3 or not parts[0].startswith("GPU-"):
            continue
        if not parts[1] or not parts[2].isdecimal():
            continue
        records.append((parts[0], parts[1], int(parts[2])))
    if len(records) != 1:
        raise RuntimeError("Require exactly one identifiable NVIDIA GPU; no guessing")
    uuid, pci, free = records[0]
    if free < 512:
        raise RuntimeError("Less than 512 MiB GPU memory free; refuse model test")
    return uuid, pci


def _parse_offload(native_stderr: str) -> tuple[int, int]:
    results = [(int(a), int(b)) for a, b in OFFLOAD_RE.findall(native_stderr)]
    if not results:
        return 0, 0
    return max(results, key=lambda x: (x[0], x[1]))


def probe_inference(
    *, project_root: Path = PROJECT_ROOT, purpose: str = "test",
    authorize_gpu_inference: bool = False,
    environment: Any = None,
    run: Callable[..., Any] = subprocess.run,
    native_verification: Callable[..., Any] = verify_native_library,
    gpu_discovery: Callable[..., tuple[str, str]] = _select_unique_gpu,
) -> GpuInferenceReport:
    root = Path(project_root)
    if not authorize_gpu_inference:
        return _report("gpu_inference_not_authorized",
                       error="Explicit --allow-gpu-inference required")
    if purpose != "test":
        return _report("production_inference_not_authorized",
                       error="Only the pinned small test GGUF is qualified for smoke")
    env = environment if environment is not None else inspect_python_environment()
    valid_venv = bool(env.interpreter_in_project_venv
                      and env.python_version.startswith("3.12.")
                      and Path(env.project_root).resolve() == root.resolve()
                      and Path(env.interpreter).resolve() ==
                          (root / ".venv/bin/python").resolve())
    if not valid_venv:
        return _report("invalid_python_venv", error="Project Python 3.12 .venv required")
    try:
        choice = select_model(purpose, project_root=root)
        valid, model, error = _valid_local_gguf(root, choice)
        if not valid:
            return _report("model_integrity_failed", model=model, error=error)
        native = native_verification(project_root=root, environment=env)
        if native.status != "native_library_dual_abi_verified":
            return _report("native_runtime_unverified", model=model, verified=True,
                           error=native.status)
        uuid, pci = gpu_discovery(run=run)
        # Binding selects main_gpu=0; the visible device is pinned by UUID.
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
                     "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        child_env.pop("LLAMA_CPP_LIB_PATH", None)
        cmd = [str(root / ".venv/bin/python"), "-I", "-c", WORKER_SCRIPT,
               str(root), str(root), purpose, uuid]
        completed = run(
            cmd, cwd=str(root), capture_output=True, text=True,
            check=False, timeout=TIMEOUT_SECONDS, env=child_env,
        )
        results = [line[len(WORKER_PREFIX):] for line in
                   (completed.stdout or "").splitlines()
                   if line.startswith(WORKER_PREFIX)]
        if not results or len(results) != 1:
            return _report("worker_failed_or_no_result", model=model, verified=True,
                           native=True, uuid=uuid, pci=pci,
                           error="Worker did not emit exactly one machine result")
        payload = json.loads(results[0])
        if not isinstance(payload, dict) or payload.get("schema") != \
                "ruth-cuda-gguf-smoke-v1":
            raise ValueError("Malformed worker schema")
        loaded = payload.get("model_loaded") is True
        tokens = payload.get("completion_tokens")
        if type(tokens) is not int or tokens < 0:
            raise ValueError("Invalid completion token count")
        observed = payload.get("gpu_process_observed") is True
        mem = payload.get("gpu_memory_peak_mib_observed")
        if mem is not None and (type(mem) is not int or mem < 0):
            raise ValueError("Invalid process memory telemetry")
        if completed.returncode != 0:
            return _report(
                "gpu_inference_worker_failed", model=model, verified=True,
                native=True, uuid=uuid, pci=pci,
                error=str(payload.get("error", "Worker exited without successful generation"))[:240],
            )
        if payload.get("gpu_uuid") != uuid or payload.get("purpose") != purpose:
            raise ValueError("Worker GPU/model identity mismatch")
        duration = payload.get("worker_duration_seconds")
        if type(duration) not in (float, int) or duration < 0:
            duration = None
        offloaded, total = _parse_offload(completed.stderr or "")
        passed = completed.returncode == 0 and loaded and tokens > 0 \
                 and offloaded > 0 and total >= offloaded and observed \
                 and mem is not None and mem > 0
        return _report(
            "gpu_gguf_inference_verified" if passed else "gpu_inference_inconclusive",
            model=model, verified=True, native=True, uuid=uuid, pci=pci,
            loaded=loaded, tokens=tokens, offloaded=offloaded, total=total,
            process=observed, memory=mem, duration=duration,
            error=None if passed else (
                "Incomplete CUDA evidence; check worker exit, layer-offload "
                "diagnostics and nvidia-smi worker PID/UUID observation"
            ),
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError,
            json.JSONDecodeError) as exc:
        return _report("gpu_inference_probe_failed",
                       error=type(exc).__name__ + ": " + str(exc)[:280])
