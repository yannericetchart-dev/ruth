"""Fail-closed GPU profile orchestration, median/min/max and JSON provenance.

One worker, one GGUF load, one excluded warmup, five identical sampling
rounds. CPU fallback forbidden. Reuses proven Phase 07 native validation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any, Callable

from .gpu_preflight import inspect_gpu_preflight
from .dependencies import PROJECT_ROOT
from .inference_probe import (
    _valid_local_gguf, _parse_offload, _select_unique_gpu,
)
from .model_catalog import select_model
from .native_library import verify_native_library
from .profiling_worker import (
    PROFILE_PREFIX, SAMPLES, WARMUP, TOKENS, CTX, SEED,
    PROFILE_FIELDS, _stats,
)
from .venv_packages import inspect_python_environment

PROFILE_COMMAND = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from ruth_runtime.profiling_worker import main;"
    "sys.exit(main(sys.argv[2:]))"
)
TIMEOUT = 480
PROFILES = Path(".ruth-build") / "profiles"


@dataclass(frozen=True)
class ProfileResult:
    status: str
    model: str | None
    gpu_uuid: str | None
    gpu_pci_bus: str | None
    model_verified: bool
    native_library_verified: bool
    samples: int
    warmup_rounds: int
    n_ctx: int
    seed: int
    max_tokens: int
    offloaded_layers: int
    total_layers: int
    gpu_process_observed: bool
    gpu_memory_peak_mib_observed: int | None
    model_load_seconds: float | None
    statistics: dict | None
    measurements: tuple[dict, ...]
    gpu_profile_verified: bool
    model_inference_executed: bool
    cpu_fallback_selected: bool
    diskcache_todo_open: bool
    saved_report: str | None
    error: str | None
    gpu_preflight: dict | None

    def to_dict(self) -> dict:
        return asdict(self)


def _empty(status: str, *, model: Path | None = None, uuid: str | None = None,
           pci: str | None = None, verified: bool = False, native: bool = False,
           error: str | None = None,
           gpu_preflight: dict | None = None) -> ProfileResult:
    return ProfileResult(
        status=status, model=str(model) if model else None,
        gpu_uuid=uuid, gpu_pci_bus=pci, model_verified=verified,
        native_library_verified=native, samples=0, warmup_rounds=WARMUP,
        n_ctx=CTX, seed=SEED, max_tokens=TOKENS,
        offloaded_layers=0, total_layers=0, gpu_process_observed=False,
        gpu_memory_peak_mib_observed=None, model_load_seconds=None,
        statistics=None, measurements=(), gpu_profile_verified=False,
        model_inference_executed=False, cpu_fallback_selected=False,
        diskcache_todo_open=True, saved_report=None, error=error, gpu_preflight=gpu_preflight,
    )


def _check_rows(payload: dict) -> tuple[tuple[dict, ...], dict]:
    if payload.get("schema") != "ruth-cuda-profile-v1" or \
            payload.get("purpose") != "test" or \
            payload.get("samples_requested") != SAMPLES or \
            payload.get("samples_completed") != SAMPLES or \
            payload.get("warmup_rounds") != WARMUP or \
            payload.get("n_ctx") != CTX or \
            payload.get("max_tokens") != TOKENS or \
            payload.get("seed") != SEED or \
            payload.get("n_gpu_layers_requested") != -1 or \
            payload.get("cache_enabled") is not False:
        raise ValueError("Invalid CUDA profiling worker configuration")
    samples = payload.get("rounds")
    if not isinstance(samples, list) or len(samples) != SAMPLES:
        raise ValueError("Missing profile samples")
    fields = ("tokenize_seconds", "prefill_seconds", "first_token_seconds",
              "decode_seconds")
    throughputs = ("prefill_tokens_per_second", "decode_tokens_per_second")
    checked: list[dict] = []
    for row in samples:
        if not isinstance(row, dict):
            raise ValueError("Invalid worker round")
        for key in ("prompt_tokens", "completion_tokens", "decode_tokens"):
            if type(row.get(key)) is not int or row[key] < 1:
                raise ValueError("Invalid generation token count")
        if row["decode_tokens"] != row["completion_tokens"] - 1 \
                or row["completion_tokens"] > TOKENS or row["decode_tokens"] < 8 \
                or row["prompt_tokens"] + TOKENS > CTX:
            raise ValueError("Inconsistent or insufficient decode length")
        for field in fields:
            if type(row.get(field)) not in (float, int) or \
                    not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("Missing positive native timing: " + field)
        for field in throughputs:
            if type(row.get(field)) not in (float, int) or \
                    not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("Missing measured throughput: " + field)
        if not math.isclose(row["prefill_tokens_per_second"],
                            row["prompt_tokens"] / row["prefill_seconds"],
                            rel_tol=1e-6) or \
                not math.isclose(row["decode_tokens_per_second"],
                                 row["decode_tokens"] / row["decode_seconds"],
                                 rel_tol=1e-6):
            raise ValueError("Inconsistent measured token/s")
        if type(row.get("eos")) is not bool:
            raise ValueError("Missing EOS marker")
        checked.append({k: row[k] for k in (*PROFILE_FIELDS, "eos")})
    # Recompute stats in parent, never trust a child's aggregate.
    stats = _stats(checked)
    for name in PROFILE_FIELDS:
        numbers = [row[name] for row in checked]
        expected = stats[name]
        if expected is None or expected["min"] != min(numbers) \
                or expected["max"] != max(numbers) \
                or expected["median"] != statistics.median(numbers):
            raise ValueError("Aggregated baseline missing")
    return tuple(checked), stats


def run_gpu_profile(
    *, project_root: Path = PROJECT_ROOT, purpose: str = "test",
    authorize_gpu_inference: bool = False,
    environment: Any = None,
    run: Callable[..., Any] = subprocess.run,
    native_check: Callable[..., Any] = verify_native_library,
    gpu_check: Callable[..., tuple[str, str]] = _select_unique_gpu,
    gpu_preflight: Callable[..., Any] = inspect_gpu_preflight,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    save: bool = True,
) -> ProfileResult:
    root = Path(project_root)
    if not authorize_gpu_inference:
        return _empty("gpu_profiling_not_authorized",
                      error="Explicit --allow-gpu-inference is required")
    if purpose != "test":
        return _empty("gpu_production_profiling_not_authorized",
                      error="Gemma production profiling not yet qualified")
    env = environment if environment is not None else inspect_python_environment()
    if not (env.interpreter_in_project_venv and env.python_version.startswith("3.12.")
            and Path(env.project_root).resolve() == root.resolve()
            and Path(env.interpreter).resolve() ==
                (root / ".venv/bin/python").resolve()):
        return _empty("invalid_project_venv")
    try:
        choice = select_model(purpose, project_root=root)
        good, file, reason = _valid_local_gguf(root, choice)
        if not good:
            return _empty("model_integrity_failed", model=file, error=reason)
        native = native_check(project_root=root, environment=env)
        if native.status != "native_library_dual_abi_verified":
            return _empty("native_runtime_unverified", model=file,
                          verified=True, error=native.status)
        uuid, pci = gpu_check(run=run)
        gate = gpu_preflight(uuid, run=run)
        if gate.status != "gpu_ready":
            return _empty(gate.status, model=file, verified=True,
                          native=True, uuid=uuid, pci=pci,
                          gpu_preflight=gate.to_dict(),
                          error=gate.error or "GPU occupata.")
        envvars = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
                   "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        envvars.pop("LLAMA_CPP_LIB_PATH", None)
        cmd = [str(root / ".venv/bin/python"), "-I", "-c",
               PROFILE_COMMAND, str(root), str(root), purpose, uuid]
        gate = gpu_preflight(uuid, run=run)
        if gate.status != "gpu_ready":
            return _empty(gate.status, model=file, verified=True,
                          native=True, uuid=uuid, pci=pci,
                          gpu_preflight=gate.to_dict(),
                          error=gate.error or "GPU occupata.")
        process = run(cmd, cwd=str(root), capture_output=True, text=True,
                      check=False, timeout=TIMEOUT, env=envvars)
        stdout = (process.stdout or "").splitlines()
        reports = [row[len(PROFILE_PREFIX):] for row in stdout
                   if row.startswith(PROFILE_PREFIX)]
        if process.returncode or len(reports) != 1:
            return _empty("gpu_profile_worker_failed", model=file,
                          verified=True, native=True, uuid=uuid, pci=pci,
                          error="Worker failed / incomplete result; no performance claims")
        payload = json.loads(reports[0])
        if not isinstance(payload, dict) or payload.get("gpu_uuid") != uuid \
                or payload.get("model_loaded") is not True \
                or type(payload.get("worker_pid")) is not int \
                or payload["worker_pid"] < 1 \
                or payload.get("gpu_process_observed") is not True:
            return _empty("gpu_profile_inconclusive", model=file, verified=True,
                          native=True, uuid=uuid, pci=pci,
                          error="No matching model and GPU PID evidence")
        peak = payload.get("gpu_memory_peak_mib_observed")
        load = payload.get("model_load_seconds")
        if type(peak) is not int or peak <= 0 or \
                type(load) not in (float, int) or not math.isfinite(load) or load <= 0:
            return _empty("gpu_profile_inconclusive", model=file, verified=True,
                          native=True, uuid=uuid, pci=pci,
                          error="Invalid GPU process memory or model loading duration")
        offload, layers = _parse_offload(process.stderr or "")
        if layers == 0 or offload != layers:
            return _empty("gpu_profile_inconclusive", model=file, verified=True,
                          native=True, uuid=uuid, pci=pci,
                          error="Full native GPU layer offload not proven")
        rows, stats = _check_rows(payload)
        result = ProfileResult(
            status="gpu_profile_verified", model=str(file),
            gpu_uuid=uuid, gpu_pci_bus=pci, model_verified=True,
            native_library_verified=True, samples=SAMPLES,
            warmup_rounds=WARMUP, n_ctx=CTX, seed=SEED, max_tokens=TOKENS,
            offloaded_layers=offload, total_layers=layers,
            gpu_process_observed=True, gpu_memory_peak_mib_observed=peak,
            model_load_seconds=load, statistics=stats, measurements=rows,
            gpu_profile_verified=True, model_inference_executed=True,
            cpu_fallback_selected=False, diskcache_todo_open=True,
            saved_report=None, error=None,
            gpu_preflight=gate.to_dict(),
        )
        if save:
            folder = root / PROFILES
            if (root / ".ruth-build").is_symlink() or folder.is_symlink():
                raise ValueError("Unsafe profile destination")
            folder.mkdir(parents=True, exist_ok=True)
            stamp = now().astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = folder / ("llama-3.2-1b-q4km-" + stamp + ".json")
            payload_result = result.to_dict()
            payload_result["model_sha256"] = choice.expected_sha256
            payload_result["model_revision"] = choice.revision
            payload_result["profile_schema"] = "ruth-cuda-profile-result-v1"
            with output.open("x", encoding="utf-8") as handle:
                json.dump(payload_result, handle, indent=2, sort_keys=True)
                handle.write("\n")
            result = ProfileResult(**{**result.to_dict(), "saved_report": str(output)})
        return result
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return _empty("gpu_profile_failed", error=type(exc).__name__ +
                      ": " + str(exc)[:280])
