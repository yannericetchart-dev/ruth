"""Read-only 18-case plan or explicitly authorized, sequential CUDA allocations.

NEVER evaluates a token. Execution delegates each case to Ruth's existing
probe-load CLI in a fresh Python process with full native/ABI, GGUF and GPU
checks. On failure, stop; never retry with an alternate dtype or backend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

from .dependencies import PROJECT_ROOT
from .gemma3_context_gate import (
    GEMMA3_CONTEXT_PAIRS, validate_gemma3_context_request,
)

BASELINE = (("off", "F16", "F16"), ("on", "BF16", "BF16"))
MATRIX = (*BASELINE, *sorted(GEMMA3_CONTEXT_PAIRS - set(BASELINE)))
if len(MATRIX) != 18 or len(set(MATRIX)) != 18:
    raise RuntimeError("Gemma3 18-case matrix is incomplete")


def matrix_cases() -> list[dict]:
    """Deterministic and read-only. First two cases match prior native probes."""
    result = []
    for i, (flash, k, v) in enumerate(MATRIX, 1):
        validate_gemma3_context_request(
            n_ctx=256, n_batch=64, n_ubatch=64, swa_full=True,
            flash_attention=flash, type_k=k, type_v=v, cuda_preinit=False)
        scalar_mib = {"F16": 3.25, "BF16": 3.25, "F32": 6.5}
        result.append({
            "case": i, "flash_attention": flash, "type_k": k, "type_v": v,
            "kv_logical_mib_source_derived": scalar_mib[k] + scalar_mib[v],
            "native_tested_by_this_run": False,
            "inference_approved": False, "production_approved": False,
        })
    return result


def command_for_case(row: dict, root: Path = PROJECT_ROOT) -> list[str]:
    """No shell, no optional runtime overrides, always local evidence only."""
    validate_gemma3_context_request(
        n_ctx=256, n_batch=64, n_ubatch=64, swa_full=True,
        flash_attention=row["flash_attention"],
        type_k=row["type_k"], type_v=row["type_v"], cuda_preinit=False)
    return [
        str(root / ".venv/bin/python"), "-m", "ruth_runtime",
        "models", "probe-load", "--purpose", "production",
        "--family", "gemma", "--model", "gemma-3",
        "--version", "gemma-3-1b-it", "--format", "gguf",
        "--quantization", "BF16", "--load-mode", "context",
        "--n-ctx", "256", "--n-batch", "64", "--n-ubatch", "64",
        "--swa-full", "on",
        "--flash-attn", row["flash_attention"],
        "--type-k", row["type_k"], "--type-v", row["type_v"],
        "--allow-gpu-load", "--no-git-sync",
    ]


def execute_matrix(
    *, cases: list[dict], project_root: Path = PROJECT_ROOT,
    run: Callable = subprocess.run,
) -> dict:
    """Execute selected explicit cases sequentially, stop at first failure.

    Full evidence is archived by probe-load, not republished here. This
    orchestrator only retains a short summary, never upgrades a context
    success to successful inference or production eligibility.
    """
    root = Path(project_root)
    expected = (root / ".venv/bin/python").resolve(strict=True)
    if Path(sys.executable).resolve(strict=True) != expected:
        raise ValueError("Execute with this project's .venv/bin/python only")
    completed = []
    stopped_at = None
    for row in cases:
        label = f"{row['case']}: Flash={row['flash_attention']} K={row['type_k']} V={row['type_v']}"
        print("Starting pinned allocation "+label, file=sys.stderr, flush=True)
        argv = command_for_case(row, root)
        try:
            proc = run(argv, cwd=str(root), capture_output=True,
                       text=True, check=False, timeout=420)
            data = json.loads(proc.stdout)
            if not isinstance(data, dict):
                raise ValueError("Non-object Ruth probe output")
            native = data.get("native_kv_log") or {}
            passed_types = all(
                isinstance(native.get(group), dict)
                and isinstance(native[group].get("native_logical_kv"), dict)
                and native[group]["native_logical_kv"].get("key_type") == row["type_k"].lower()
                and native[group]["native_logical_kv"].get("value_type") == row["type_v"].lower()
                for group in ("global_kv", "swa_kv")
            )
            expected_success = (
                proc.returncode == 0
                and data.get("candidate") == "gemma-3-1b-it/gguf/BF16"
                and data.get("status") == "gpu_load_only_completed"
                and data.get("mode") == "context"
                and data.get("context_initialized") is True
                and data.get("full_gpu_offload_observed") is True
                and data.get("cpu_fallback_selected") is False
                and data.get("inference_executed") is False
                and data.get("generated_tokens") == 0
                and passed_types
                and native.get("n_ctx_effective_native") == 256
                and native.get("global_kv", {}).get("cells") == 256
                and native.get("swa_kv", {}).get("cells") == 256
                and (native.get("global_kv", {}).get("native_logical_kv") or {}).get("layers") == 4
                and (native.get("swa_kv", {}).get("native_logical_kv") or {}).get("layers") == 22
                and data.get("profile_recording_status") in (
                    "local_profile_updated", "local_profile_already_recorded")
                and isinstance(data.get("local_evidence_file"), str)
                and (data.get("git_publication") or {}).get("status") == "git_sync_disabled"
                and (data.get("context_options_requested") or {}).get("flash_attention")
                    == row["flash_attention"]
                and (data.get("context_options_requested") or {}).get("type_k")
                    == row["type_k"]
                and (data.get("context_options_requested") or {}).get("type_v")
                    == row["type_v"]
            )
            buffers = data.get("native_buffers") or []
            measured_kv_mib = sum(
                float(buf["mib"]) for buf in buffers if isinstance(buf, dict)
                and buf.get("kind") == "kv"
                and str(buf.get("device","")).startswith("CUDA")
                and isinstance(buf.get("mib"), (int, float))
            )
            result = {
                "case": row["case"], "flash_attention": row["flash_attention"],
                "type_k": row["type_k"], "type_v": row["type_v"],
                "status": "native_context_verified" if expected_success else
                          "native_context_failed_or_unverified",
                "probe_status": data.get("status"),
                "context_initialized": data.get("context_initialized") is True,
                "gpu_offload_verified": data.get("full_gpu_offload_observed") is True,
                "native_kv_types_verified": passed_types,
                "native_kv_buffer_mib_observed": round(measured_kv_mib, 4),
                "compute_buffers_mib_observed": [
                    buf.get("mib") for buf in buffers if isinstance(buf, dict)
                    and buf.get("kind") == "compute"
                    and str(buf.get("device","")).startswith("CUDA")
                ],
                "local_evidence_file": data.get("local_evidence_file"),
                "local_profile_recording_status": data.get("profile_recording_status"),
                "error": None if expected_success else
                         (data.get("error") or "Unverified native KV/worker result"),
                "inference_executed": False, "production_approved": False,
            }
        except (ValueError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
            result = {
                "case": row["case"], "flash_attention": row["flash_attention"],
                "type_k": row["type_k"], "type_v": row["type_v"],
                "status": "worker_exception",
                "error": type(exc).__name__ + ": " + str(exc)[:230],
                "inference_executed": False, "production_approved": False,
            }
            expected_success = False
        completed.append(result)
        print("Result: "+result["status"], file=sys.stderr, flush=True)
        if not expected_success:
            stopped_at = row["case"]
            break
    return {
        "action": "gemma3_context_matrix",
        "status": "selected_allocations_completed" if stopped_at is None
                  else "stopped_after_unverified_or_failed_case",
        "cases_requested": len(cases),
        "cases_run": len(completed),
        "stopped_at": stopped_at,
        "results": completed,
        "gpu_probe_invocations": len(completed),
        "context_allocations_observed": sum(
            result.get("context_initialized") is True for result in completed),
        "inference_executed": False,
        "production_approved": False,
        "git_sync_disabled": True,
    }


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gemma3 pinned 18-case KV allocation matrix")
    parser.add_argument("--execute", action="store_true",
                        help="Actually allocate GPU context; requires --allow-gpu-load")
    parser.add_argument("--allow-gpu-load", action="store_true",
                        help="Explicitly authorize allocation only, no inference")
    parser.add_argument("--case", type=int, choices=range(1,19), metavar="1..18",
                        help="Execute or plan one numbered case")
    parser.add_argument("--start-at", type=int, choices=range(1,19), metavar="1..18",
                        help="For resumable sequential execution; skipped cases are NOT verified")
    parsed = parser.parse_args(args)
    if parsed.case and parsed.start_at:
        parser.error("Use --case OR --start-at")
    if parsed.execute and not parsed.allow_gpu_load:
        parser.error("GPU allocation requires explicit --allow-gpu-load")
    if not parsed.execute and parsed.allow_gpu_load:
        parser.error("--allow-gpu-load is only meaningful with --execute")
    cases = matrix_cases()
    if parsed.case:
        cases = [row for row in cases if row["case"] == parsed.case]
    if parsed.start_at:
        cases = [row for row in cases if row["case"] >= parsed.start_at]
    if not parsed.execute:
        print(json.dumps({
            "action": "gemma3_context_matrix", "status": "read_only_plan",
            "cases": cases, "gpu_allocations_executed": False,
            "inference_executed": False, "git_sync_disabled": True,
        }, sort_keys=True, indent=2))
        return 0
    try:
        report = execute_matrix(cases=cases)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        print(json.dumps({
            "action": "gemma3_context_matrix",
            "status": "refused_before_gpu_worker", "error": str(exc)[:250],
            "inference_executed": False, "production_approved": False,
        }, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["status"] == "selected_allocations_completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
