"""Phase28 experiment controller. Default is read-only; GPU requires explicit opt-in.

Preflight uses Ruth's exact native ABI/GGUF/GPU gates. One selected context and
precision at a time; no fallback, no download, no automatic escalation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .gemma3_phase28_gate import (
    verify_plan, plans, theoretical_kv_mib, REPEATS, WARMUPS
)
from .gemma3_phase28_worker import SCHEMA, PREFIX
from .gpu_preflight import inspect_gpu_preflight
from .inference_probe import _valid_local_gguf, _select_unique_gpu, _parse_offload
from .llama_kv_log import parse_llama_context_log
from .load_only_probe import extract_native_buffers
from .model_catalog import select_model
from .native_library import verify_native_library
from .venv_packages import inspect_python_environment
from .context_source_evidence import pinned_native_wheel_sha256

WORKER_SCRIPT = (
    "import sys;sys.path.insert(0,sys.argv[1]);"
    "from ruth_runtime.gemma3_phase28_worker import main;"
    "sys.exit(main(sys.argv[2:]))"
)
# Independent conservative free-memory floor; not inferred to be exact
# safe VRAM for larger contexts. Native failures still halt the experiment.
MIN_FREE_VRAM_MIB = 3072
TIMEOUT = 900


def verify_artifact(root: Path, record: object, n_ctx: int) -> bool:
    if not isinstance(record, dict):
        return False
    name, sha = record.get("relative_path"), record.get("sha256")
    if (not isinstance(name, str) or not isinstance(sha, str)
        or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
        return False
    relative = Path(name)
    if (relative.is_absolute() or ".." in relative.parts
        or len(relative.parts) != 4
        or relative.parts[:3] != (".ruth-build","phase28","gemma-3-1b-it")
        or not relative.name.startswith(f"ctx-{n_ctx}_")
        or relative.suffix != ".npz"):
        return False
    target = root / relative
    if (target.is_symlink() or not target.is_file()
        or not 0 < target.stat().st_size <= 40_000_000):
        return False
    h = hashlib.sha256()
    with target.open("rb") as f:
        for blob in iter(lambda: f.read(1024*1024), b""):
            h.update(blob)
    return h.hexdigest() == sha


def has_memory_reserve(preflight) -> bool:
    free = getattr(preflight, "memory_mib", None)
    return (
        getattr(preflight, "status", None) == "gpu_ready"
        and isinstance(free, dict)
        and type(free.get("free")) is int
        and free["free"] >= MIN_FREE_VRAM_MIB
    )


def probe_one(
    candidate: str, n_ctx: int, *,
    authorize_gpu_inference: bool = False,
    project_root: Path = PROJECT_ROOT,
    environment: Any = None,
    run: Callable = subprocess.run,
    verify_native: Callable = verify_native_library,
    discover_gpu: Callable = _select_unique_gpu,
    gpu_preflight: Callable = inspect_gpu_preflight,
) -> dict:
    base = {
        "candidate": candidate, "n_ctx_requested": n_ctx,
        "inference_executed": False, "native_kernel_dispatch_verified": False,
        "production_qualified": False, "runtime_profile_published": False,
    }
    def finish(status, **more):
        return {**base, "status": status, **more}
    if not authorize_gpu_inference:
        return finish("gpu_inference_not_authorized")
    try:
        plan = verify_plan(candidate, n_ctx)
    except (RuntimeError, ValueError):
        return finish("unapproved_extended_context")
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    if not (
        env.interpreter_in_project_venv
        and env.python_version.startswith("3.12.")
        and Path(env.project_root).resolve() == root.resolve()
        and Path(env.interpreter).resolve() ==
            (root / ".venv/bin/python").resolve()
    ):
        return finish("invalid_project_venv")
    try:
        choice = select_model(
            "production", family="gemma", model="gemma-3",
            version="gemma-3-1b-it", format="gguf",
            quantization="BF16", project_root=root)
        valid, _path, reason = _valid_local_gguf(root, choice)
        if not valid:
            return finish("pinned_gguf_unverified", error=reason)
        library = verify_native(project_root=root, environment=env)
        if library.status != "native_library_dual_abi_verified":
            return finish("native_library_unverified", error=library.status)
        wheel_sha = pinned_native_wheel_sha256(root)
        uuid, pci = discover_gpu(run=run)
        first = gpu_preflight(uuid, run=run)
        if not has_memory_reserve(first):
            return finish("gpu_not_exclusive_or_insufficient_free_vram",
                          free_mib=(first.memory_mib or {}).get("free"),
                          required_free_mib=MIN_FREE_VRAM_MIB)
        child_env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid,
                     "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        child_env.pop("LLAMA_CPP_LIB_PATH", None)
        child_env.pop("RUTH_LLAMA_KV_OPTIONS", None)
        cmd = [str(root / ".venv/bin/python"), "-I", "-c", WORKER_SCRIPT,
               str(root), str(root), uuid, candidate, str(n_ctx)]
        second = gpu_preflight(uuid, run=run)
        if not has_memory_reserve(second):
            return finish("gpu_preflight_changed",
                          free_mib=(second.memory_mib or {}).get("free"),
                          required_free_mib=MIN_FREE_VRAM_MIB)
        result = run(cmd, cwd=str(root), env=child_env, text=True,
                     capture_output=True, check=False, timeout=TIMEOUT)
        lines = [line[len(PREFIX):] for line in (result.stdout or "").splitlines()
                 if line.startswith(PREFIX)]
        if len(lines) != 1:
            return finish("worker_missing_unique_native_result",
                          inference_executed=None,
                          error=(result.stderr or "")[-240:])
        data = json.loads(lines[0])
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError("Unknown Phase28 worker JSON schema")
        if result.returncode != 0:
            return finish("extended_context_inference_failed",
                          inference_executed=data.get("inference_executed"),
                          error=data.get("error"))
        layer, total = _parse_offload(result.stderr or "")
        native = parse_llama_context_log(result.stderr or "")
        kv_ok = all(
            isinstance(native.get(group), dict)
            and native[group].get("cells") == plan.n_ctx
            and isinstance(native[group].get("native_logical_kv"),dict)
            and native[group]["native_logical_kv"].get("layers") == count
            and native[group]["native_logical_kv"].get("key_type") ==
                plan.type_k.lower()
            and native[group]["native_logical_kv"].get("value_type") ==
                plan.type_v.lower()
            for group, count in (("global_kv",4),("swa_kv",22))
        )
        timing = data.get("timings")
        timing_ok = (
            isinstance(timing,list) and len(timing)==REPEATS
            and all(
                isinstance(t,dict)
                and type(t.get("prefill_ms")) in (float,int)
                and math.isfinite(t["prefill_ms"]) and t["prefill_ms"]>0
                and t.get("prefill_chunks") ==
                    (plan.prompt_tokens+plan.n_batch-1)//plan.n_batch
                and isinstance(t.get("decode_ms_per_token"),list)
                and len(t["decode_ms_per_token"]) == plan.decode_tokens
                and all(type(x) in (float,int) and math.isfinite(x) and x>0
                        for x in t["decode_ms_per_token"])
                for t in timing)
        )
        expected = (
            data.get("status") == "long_context_teacher_forced_complete"
            and data.get("model_version") == "gemma-3-1b-it"
            and data.get("quantization") == "BF16"
            and data.get("model_sha256") == choice.expected_sha256
            and data.get("gpu_uuid") == uuid
            and data.get("candidate") == plan.candidate
            and data.get("n_ctx") == plan.n_ctx
            and data.get("n_batch") == plan.n_batch
            and data.get("n_ubatch") == plan.n_ubatch
            and data.get("swa_full") is True
            and data.get("flash_attention") == plan.flash_attention
            and data.get("type_k") == plan.type_k
            and data.get("type_v") == plan.type_v
            and data.get("prefill_tokens") == plan.prompt_tokens
            and data.get("decode_tokens") == plan.decode_tokens
            and data.get("decode_logits_requested_per_token") is True
            and type(data.get("decode_logits_requested_count")) is int
            and data.get("decode_logits_requested_count") == plan.decode_tokens
            and data.get("warmups") == WARMUPS
            and data.get("repeats") == REPEATS
            and data.get("generated_tokens") == 0
            and data.get("inference_executed") is True
            and data.get("production_qualified") is False
            and data.get("native_kernel_dispatch_verified") is False
            and data.get("gpu_process_observed") is True
            and layer == total and layer == 27
            and native.get("n_ctx_effective_native") == plan.n_ctx
            and native.get("n_ctx_seq_effective_native") == plan.n_ctx
            and native.get("n_batch_effective_native") == plan.n_batch
            and native.get("n_ubatch_effective_native") == plan.n_ubatch
            and native.get("n_seq_max_native") == 1
            and native.get("flash_attention_native_parameter") ==
                {"off":"disabled","on":"enabled"}[plan.flash_attention]
            and kv_ok and timing_ok
            and verify_artifact(root,data.get("logits_artifact"),plan.n_ctx)
        )
        if not expected:
            return finish("extended_context_native_evidence_unverified",
                          inference_executed=data.get("inference_executed"),
                          error="Offload, dtype, native geometry, timing or local logits mismatch")
        prefill = [t["prefill_ms"] for t in timing]
        decode = [ms for t in timing for ms in t["decode_ms_per_token"]]
        median_prefill = statistics.median(prefill)
        median_decode = statistics.median(decode)
        return finish(
            "native_long_context_inference_observed",
            inference_executed=True, full_gpu_offload_observed=True,
            gpu_uuid=uuid, gpu_pci_bus=pci,
            model_sha256=choice.expected_sha256, native_wheel_sha256=wheel_sha,
            flash_attention=plan.flash_attention,
            type_k=plan.type_k, type_v=plan.type_v,
            n_ctx_effective_native=plan.n_ctx,
            prompt_tokens=plan.prompt_tokens, decode_tokens=plan.decode_tokens,
            decode_logits_requested_per_token=True,
            decode_logits_requested_count=plan.decode_tokens,
            prompt_ids_sha256=data.get("prompt_ids_sha256"),
            decode_ids=data.get("decode_ids"),
            prefill_ms_median=round(median_prefill,4),
            prefill_tokens_per_second=round(plan.prompt_tokens*1000/median_prefill,2),
            decode_ms_per_token_median=round(median_decode,4),
            decode_tokens_per_second=round(1000/median_decode,2),
            sampled_gpu_peak_mib=data.get("gpu_memory_peak_mib_sampled"),
            nvtx_ranges_available=data.get("nvtx_ranges_available") is True,
            native_buffers=extract_native_buffers(result.stderr or ""),
            kv_logical_mib_source_derived=theoretical_kv_mib(plan),
            logits_artifact=data["logits_artifact"],
            timings=timing, offloaded_layers=layer,total_layers=total,
        )
    except (OSError,RuntimeError,ValueError,TypeError,KeyError,
            subprocess.SubprocessError,json.JSONDecodeError) as exc:
        return finish("phase28_probe_failed", inference_executed=None,
                      error=type(exc).__name__+": "+str(exc)[:220])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pinned Gemma3 Phase28 long-context GPU inference (opt-in)")
    parser.add_argument("--candidate",choices=[
        "f16-off","bf16-on","f32-off","mixed-on"])
    parser.add_argument("--n-ctx",type=int,choices=[256,512,1024,2048])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-gpu-inference", action="store_true")
    opts = parser.parse_args(argv)
    if (opts.candidate is None) != (opts.n_ctx is None):
        parser.error("--candidate and --n-ctx must both be provided")
    if opts.execute and not opts.allow_gpu_inference:
        parser.error("Explicit --allow-gpu-inference required")
    if opts.allow_gpu_inference and not opts.execute:
        parser.error("--allow-gpu-inference requires --execute")
    if opts.execute and opts.candidate is None:
        parser.error("Select one candidate and context explicitly")
    if not opts.execute:
        choices = [p.as_dict() | {"kv_logical_mib_source_derived":theoretical_kv_mib(p)}
                   for p in plans()
                   if (opts.candidate is None or
                       (opts.candidate == p.candidate and opts.n_ctx == p.n_ctx))]
        print(json.dumps({
            "action":"gemma3_phase28","status":"read_only_plan",
            "cases":choices,"inference_executed":False,
            "production_qualified":False,
        },sort_keys=True,indent=2))
        return 0
    outcome = probe_one(
        opts.candidate,opts.n_ctx,authorize_gpu_inference=True)
    print(json.dumps({
        "action":"gemma3_phase28", **outcome,
    },sort_keys=True,indent=2))
    return 0 if outcome["status"]=="native_long_context_inference_observed" else 2


if __name__=="__main__":
    raise SystemExit(main())
