"""Gemma3 Phase27: read-only plan or explicitly authorized native inference.

Fixed teacher-forced tokens, no text generation, no Git publication,
automatic cache choice, fallback or production qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Callable, Any

from .dependencies import PROJECT_ROOT
from .gemma3_context_matrix import matrix_cases
from .gemma3_context_gate import validate_gemma3_context_request
from .gemma3_benchmark_worker import (
    SCHEMA, PREFIX, REPEATS, WARMUPS, DECODE_STEPS,
)
from .gpu_preflight import inspect_gpu_preflight
from .inference_probe import _valid_local_gguf, _select_unique_gpu, _parse_offload
from .install_plan import WHEEL_VERSION
from .model_catalog import select_model
from .native_library import verify_native_library
from .context_source_evidence import pinned_native_wheel_sha256
from .venv_packages import inspect_python_environment

TIMEOUT = 600
WORKER_SCRIPT = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from ruth_runtime.gemma3_benchmark_worker import main;"
    "sys.exit(main(sys.argv[2:]))"
)


def verify_logits_artifact(root: Path, value: object) -> bool:
    if not isinstance(value, dict):
        return False
    rel = value.get("relative_path")
    sha = value.get("sha256")
    if (not isinstance(rel, str) or not isinstance(sha, str)
        or len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha)):
        return False
    path = Path(rel)
    if (path.is_absolute() or ".." in path.parts or
        len(path.parts) != 4 or path.parts[:3] !=
            (".ruth-build", "phase27", "gemma-3-1b-it") or
        not path.name.endswith(".npz")):
        return False
    fp = root / path
    if fp.is_symlink() or not fp.is_file() or fp.stat().st_size > 150_000_000:
        return False
    h = hashlib.sha256()
    with fp.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest() == sha


def probe_one(
    row: dict, *, authorize_gpu_inference: bool = False,
    project_root: Path = PROJECT_ROOT,
    environment: Any = None,
    run: Callable = subprocess.run,
    verify_native: Callable = verify_native_library,
    discover_gpu: Callable = _select_unique_gpu,
    gpu_preflight: Callable = inspect_gpu_preflight,
) -> dict:
    base = {
        "case": row.get("case"), "type_k": row.get("type_k"),
        "type_v": row.get("type_v"), "flash_attention": row.get("flash_attention"),
        "inference_executed": False, "production_qualified": False,
        "native_kernel_dispatch_verified": False,
        "runtime_profile_published": False,
    }
    def finish(status, **kw):
        return {**base, "status": status, **kw}
    if not authorize_gpu_inference:
        return finish("gpu_inference_not_authorized")
    try:
        flash, k, v = row["flash_attention"], row["type_k"], row["type_v"]
        validate_gemma3_context_request(
            n_ctx=256, n_batch=64, n_ubatch=64, swa_full=True,
            flash_attention=flash, type_k=k, type_v=v, cuda_preinit=False)
        if row not in matrix_cases():
            raise ValueError("Not a canonical Phase27 matrix case")
    except (TypeError, KeyError, ValueError) as exc:
        return finish("unapproved_benchmark_configuration",
                      error=str(exc)[:190])
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    if not (env.interpreter_in_project_venv
            and env.python_version.startswith("3.12.")
            and Path(env.project_root).resolve() == root.resolve()
            and Path(env.interpreter).resolve() ==
                (root / ".venv/bin/python").resolve()):
        return finish("invalid_project_venv")
    try:
        choice = select_model(
            "production", family="gemma", model="gemma-3",
            version="gemma-3-1b-it", quantization="BF16",
            format="gguf", project_root=root)
        valid, _, reason = _valid_local_gguf(root, choice)
        if not valid:
            return finish("pinned_gguf_unverified", error=reason)
        native = verify_native(project_root=root, environment=env)
        if native.status != "native_library_dual_abi_verified":
            return finish("native_wheel_unverified", error=native.status)
        wheel_sha = pinned_native_wheel_sha256(root)
        uuid, pci = discover_gpu(run=run)
        pre = gpu_preflight(uuid, run=run)
        if pre.status != "gpu_ready":
            return finish("gpu_preflight_failed", error=pre.status)
        worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
                      "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        worker_env.pop("LLAMA_CPP_LIB_PATH", None)
        worker_env.pop("RUTH_LLAMA_KV_OPTIONS", None)
        args = [str(root / ".venv/bin/python"), "-I", "-c", WORKER_SCRIPT,
                str(root), str(root), uuid, flash, k, v]
        second = gpu_preflight(uuid, run=run)
        if second.status != "gpu_ready":
            return finish("gpu_preflight_changed", error=second.status)
        proc = run(args, cwd=str(root), env=worker_env,
                   capture_output=True, text=True, check=False, timeout=TIMEOUT)
        lines = [s[len(PREFIX):] for s in (proc.stdout or "").splitlines()
                 if s.startswith(PREFIX)]
        if len(lines) != 1:
            return finish("worker_missing_unique_result",
                          error=(proc.stderr or "")[-220:])
        data = json.loads(lines[0])
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError("Unrecognized native benchmark result")
        if proc.returncode != 0:
            return finish("native_inference_failed_or_unsupported",
                          error=data.get("error"),
                          inference_executed=data.get("inference_executed"))
        layers, total = _parse_offload(proc.stderr or "")
        timing = data.get("timings")
        durations_valid = (
            isinstance(timing, list) and len(timing) == REPEATS
            and all(
                isinstance(t, dict)
                and type(t.get("prefill_ms")) in (float, int)
                and math.isfinite(t["prefill_ms"]) and t["prefill_ms"] > 0
                and isinstance(t.get("decode_ms_per_token"), list)
                and len(t["decode_ms_per_token"]) == DECODE_STEPS
                and all(type(x) in (float, int) and math.isfinite(x) and x > 0
                        for x in t["decode_ms_per_token"])
                for t in timing)
        )
        expected = (
            data.get("status") == "teacher_forced_gpu_workload_complete"
            and data.get("model_sha256") == choice.expected_sha256
            and data.get("model_version") == "gemma-3-1b-it"
            and data.get("quantization") == "BF16"
            and (data.get("flash_attention"), data.get("type_k"), data.get("type_v"))
                == (flash, k, v)
            and data.get("gpu_uuid") == uuid
            and data.get("n_ctx") == 256
            and data.get("n_batch") == 64
            and data.get("n_ubatch") == 64
            and data.get("inference_executed") is True
            and data.get("generated_tokens") == 0
            and data.get("native_kernel_dispatch_verified") is False
            and data.get("production_qualified") is False
            and data.get("warmups") == WARMUPS
            and data.get("repeats") == REPEATS
            and data.get("decode_tokens") == DECODE_STEPS
            and data.get("gpu_process_observed") is True
            and layers > 0 and layers == total
            and durations_valid
            and verify_logits_artifact(root, data.get("artifact"))
        )
        if not expected:
            return finish("native_benchmark_result_unverified",
                          error="Invalid worker evidence, logits file or GPU offload",
                          inference_executed=data.get("inference_executed"))
        import statistics
        prefill = [t["prefill_ms"] for t in timing]
        decoded = [x for t in timing for x in t["decode_ms_per_token"]]
        return finish(
            "native_teacher_forced_inference_observed",
            inference_executed=True, full_gpu_offload_observed=True,
            gpu_uuid=uuid, gpu_pci_bus=pci,
            gguf_sha256=choice.expected_sha256,
            native_wheel_sha256=wheel_sha, offloaded_layers=layers,
            total_layers=total,
            n_vocab=data["n_vocab"],
            prompt_tokens=data["prefill_tokens"],
            prompt_ids_sha256=data["prompt_ids_sha256"],
            decode_ids=data["decode_ids"],
            prefill_ms_median=round(statistics.median(prefill), 4),
            decode_ms_per_token_median=round(statistics.median(decoded), 4),
            timings=timing,
            gpu_peak_mib_sampled=data.get("gpu_memory_peak_mib_observed"),
            logits_artifact=data["artifact"],
            numeric_fidelity_verified=False,
            native_kernel_dispatch_verified=False,
            production_qualified=False,
        )
    except (OSError, RuntimeError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return finish("phase27_probe_failed",
                      error=type(exc).__name__ + ": " + str(exc)[:235])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gemma3 Phase27 fixed-token native prefill/decode measurements")
    parser.add_argument("--case", type=int, choices=range(1, 19), metavar="1..18",
                        help="Execute one case; read-only plan without --execute")
    parser.add_argument("--start-at", type=int, choices=range(1, 19), metavar="1..18",
                        help="Explicit sequential remaining cases; stop at first failure")
    parser.add_argument("--execute", action="store_true",
                        help="Perform real GPU prefill/decode, not context-only")
    parser.add_argument("--allow-gpu-inference", action="store_true",
                        help="Explicit authorization for deterministic CUDA model evaluation")
    opts = parser.parse_args(argv)
    if opts.case and opts.start_at:
        parser.error("Specify --case OR --start-at")
    if opts.execute and not opts.allow_gpu_inference:
        parser.error("Explicit --allow-gpu-inference is required")
    if opts.allow_gpu_inference and not opts.execute:
        parser.error("--allow-gpu-inference requires --execute")
    cases = matrix_cases()
    if opts.case:
        cases = [x for x in cases if x["case"] == opts.case]
    if opts.start_at:
        cases = [x for x in cases if x["case"] >= opts.start_at]
    if not opts.execute:
        print(json.dumps({"action": "gemma3_phase27",
                          "status": "read_only_inference_plan",
                          "cases": cases, "inference_executed": False,
                          "production_qualified": False},
                         sort_keys=True, indent=2))
        return 0
    results = []
    for row in cases:
        print(f"GPU inference case {row['case']}: Flash={row['flash_attention']} "
              f"K={row['type_k']} V={row['type_v']}", file=__import__("sys").stderr,
              flush=True)
        result = probe_one(row, authorize_gpu_inference=True)
        results.append(result)
        print(f"Status: {result['status']}", file=__import__("sys").stderr,
              flush=True)
        if result["status"] != "native_teacher_forced_inference_observed":
            break
    print(json.dumps({"action": "gemma3_phase27",
                      "status": "completed_requested_cases" if len(results) == len(cases)
                       else "stopped_at_unverified_case",
                      "cases_requested": len(cases), "cases_run": len(results),
                      "results": results, "production_qualified": False,
                      "runtime_profile_published": False},
                     sort_keys=True, indent=2))
    return 0 if len(results) == len(cases) and all(
        x["status"] == "native_teacher_forced_inference_observed"
        for x in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
