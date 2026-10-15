"""Isolated native Phase28 context-scaling worker: actual GPU inference.

Never imports external model paths, samples chat text, downloads models or
updates production profiles. Parent validates GGUF SHA, ABI, GPU and VRAM;
child repeats pinned model, GPU UUID and exact-context parameter checks.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Event, Thread
from time import monotonic_ns

from .gemma3_benchmark_worker import PROMPT, CONTINUATION, tokenizer_ids
from .gemma3_nvtx import NativeRanges
from .gemma3_phase28_gate import verify_plan, WARMUPS, REPEATS
from .inference_worker import measure_worker_gpu
from .install_plan import WHEEL_VERSION
from .load_only_worker import native_kv_dtype
from .model_catalog import select_model
from .model_download import _local_path

SCHEMA = "ruth-gemma3-phase28-worker-v1"
PREFIX = "RUTH_GEMMA3_PHASE28_RESULT="


def fixed_tokens(api, vocab, plan) -> tuple[list[int], list[int]]:
    seed = tokenizer_ids(api, vocab, PROMPT, special=True)
    filler = tokenizer_ids(api, vocab, CONTINUATION, special=False)
    if not 1 <= len(seed) < plan.prompt_tokens or len(filler) < 6:
        raise RuntimeError("Pinned Phase28 tokenizer output changed")
    prompt = (seed + (filler * (plan.prompt_tokens // len(filler) + 1)))
    prompt = prompt[:plan.prompt_tokens]
    follow = (filler * (plan.decode_tokens // len(filler) + 1))[:plan.decode_tokens]
    if len(prompt) + len(follow) >= plan.n_ctx:
        raise RuntimeError("Phase28 deterministic workload exceeds context")
    return prompt, follow


def run_workload(api, model, ctx, plan) -> dict:
    import numpy as np
    vocab = api.llama_model_get_vocab(model)
    if not vocab:
        raise RuntimeError("Native vocabulary missing")
    n_vocab = api.llama_vocab_n_tokens(vocab)
    if not 512 <= n_vocab <= 300000:
        raise RuntimeError("Unverified native vocabulary size")
    prompt, follow = fixed_tokens(api, vocab, plan)
    mem = api.llama_get_memory(ctx)
    if not mem:
        raise RuntimeError("Native KV memory unavailable")
    batch = api.llama_batch_init(plan.n_batch, 0, 1)
    if not batch:
        raise RuntimeError("Native batch allocation failed")
    timings = []
    snapshots = None
    nvtx = NativeRanges("PHASE28")
    try:
        for trial in range(WARMUPS + REPEATS):
            api.llama_memory_clear(mem, True)
            prefill_ns = 0
            decode_ns = []
            prefill_logits = None
            final_logits = None
            for offset in range(0, len(prompt), plan.n_batch):
                chunk = prompt[offset:offset + plan.n_batch]
                batch.n_tokens = len(chunk)
                for i, token in enumerate(chunk):
                    batch.token[i] = token
                    batch.pos[i] = offset + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = int(
                        offset + i == len(prompt) - 1)
                with nvtx.range("prefill"):
                    t0 = monotonic_ns()
                    ret = api.llama_decode(ctx, batch)
                    if ret != 0:
                        raise RuntimeError(f"Native prefill failed: {ret}")
                    api.llama_synchronize(ctx)
                    prefill_ns += monotonic_ns() - t0
                if offset + len(chunk) == len(prompt) and trial == WARMUPS + REPEATS - 1:
                    ptr = api.llama_get_logits_ith(ctx, -1)
                    if not ptr:
                        raise RuntimeError("Missing prefill logits")
                    prefill_logits = np.ctypeslib.as_array(ptr,shape=(n_vocab,)).copy()
            for i, token in enumerate(follow):
                batch.n_tokens = 1
                batch.token[0] = token
                batch.pos[0] = len(prompt) + i
                batch.n_seq_id[0] = 1
                batch.seq_id[0][0] = 0
                # Require the output projection at EVERY decode position.
                # Otherwise the benchmark silently times hidden-state-only
                # passes for 15/16 tokens and underreports decoding latency.
                batch.logits[0] = 1
                with nvtx.range("decode"):
                    t0 = monotonic_ns()
                    ret = api.llama_decode(ctx, batch)
                    if ret != 0:
                        raise RuntimeError(f"Native decode failed: {ret}")
                    api.llama_synchronize(ctx)
                    decode_ns.append(monotonic_ns() - t0)
                if i == len(follow) - 1 and trial == WARMUPS + REPEATS - 1:
                    ptr = api.llama_get_logits_ith(ctx, -1)
                    if not ptr:
                        raise RuntimeError("Missing final decode logits")
                    final_logits = np.ctypeslib.as_array(ptr,shape=(n_vocab,)).copy()
            if prefill_ns <= 0 or any(x <= 0 for x in decode_ns):
                raise RuntimeError("Nonpositive native timed evaluation")
            if trial >= WARMUPS:
                timings.append({
                    "prefill_ms": prefill_ns / 1e6,
                    "prefill_chunks": (len(prompt) + plan.n_batch - 1) // plan.n_batch,
                    "decode_ms_per_token": [x / 1e6 for x in decode_ns],
                })
            if trial == WARMUPS + REPEATS - 1:
                if prefill_logits is None or final_logits is None:
                    raise RuntimeError("Missing pinned logits snapshots")
                snapshots = np.stack((prefill_logits, final_logits))
    finally:
        api.llama_batch_free(batch)
    if (snapshots is None or snapshots.shape != (2, n_vocab)
        or snapshots.dtype != np.float32 or not np.isfinite(snapshots).all()
        or len(timings) != REPEATS):
        raise RuntimeError("Unverified long-context logits or timings")
    return {
        "logits": snapshots, "timings": timings,
        "prompt": prompt, "decode_ids": follow, "n_vocab": n_vocab,
        "nvtx_ranges_available": nvtx.available,
        "decode_logits_requested_per_token": True,
        "decode_logits_requested_count": len(follow),
    }


def store_logits(root: Path, plan, data: dict) -> dict:
    import numpy as np
    base = root / ".ruth-build" / "phase28" / "gemma-3-1b-it"
    if any(p.is_symlink() for p in (base.parent.parent,base.parent,base)):
        raise RuntimeError("Refusing symlink Phase28 output")
    base.mkdir(parents=True, exist_ok=True)
    fd, filepath = tempfile.mkstemp(
        dir=base, suffix=".npz",
        prefix=f"ctx-{plan.n_ctx}_flash-{plan.flash_attention}_k-{plan.type_k}_v-{plan.type_v}_")
    path = Path(filepath)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(
                out, logits=data["logits"],
                prompt_ids=np.asarray(data["prompt"], dtype=np.int32),
                decode_ids=np.asarray(data["decode_ids"], dtype=np.int32))
        h=hashlib.sha256()
        with path.open("rb") as f:
            for blob in iter(lambda: f.read(1024*1024),b""):
                h.update(blob)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {
        "relative_path": str(path.relative_to(root)), "sha256": h.hexdigest(),
        "logits_shape": list(data["logits"].shape),
        "logits_dtype": "float32",
    }


def run_native(root: Path, uuid: str, candidate: str, n_ctx: int) -> dict:
    plan = verify_plan(candidate, n_ctx)
    root = root.resolve(strict=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("Native worker GPU UUID mismatch")
    choice = select_model(
        "production", family="gemma", model="gemma-3",
        version="gemma-3-1b-it", format="gguf", quantization="BF16",
        project_root=root)
    file = _local_path(root, choice)
    if (file.is_symlink() or not file.is_file()
        or file.stat().st_size != choice.expected_size_bytes):
        raise RuntimeError("Pinned GGUF missing or wrong size")
    with file.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise RuntimeError("Pinned GGUF header invalid")
    from llama_cpp import llama_cpp as api
    from llama_cpp import __version__ as native_version
    if native_version != WHEEL_VERSION or not api.llama_supports_gpu_offload():
        raise RuntimeError("Pinned CUDA binding mismatch")
    event = Event()
    samples = []
    gpuwatch = Thread(target=measure_worker_gpu, args=(os.getpid(),uuid,event,samples),
                      daemon=True)
    gpuwatch.start()
    initialized = False
    model = None
    context = None
    try:
        api.llama_backend_init()
        initialized = True
        params = api.llama_model_default_params()
        params.n_gpu_layers = 2**31 - 1
        params.main_gpu = 0
        params.use_mmap = True
        model = api.llama_model_load_from_file(os.fsencode(file), params)
        if not model:
            raise RuntimeError("Pinned native model loading failed")
        cp = api.llama_context_default_params()
        cp.n_ctx = plan.n_ctx
        cp.n_batch = plan.n_batch
        cp.n_ubatch = plan.n_ubatch
        cp.n_seq_max = 1
        cp.kv_unified = False
        cp.swa_full = True
        cp.offload_kqv = True
        cp.type_k = native_kv_dtype(api, plan.type_k)
        cp.type_v = native_kv_dtype(api, plan.type_v)
        cp.flash_attn_type = (
            api.LLAMA_FLASH_ATTN_TYPE_ENABLED if plan.flash_attention == "on"
            else api.LLAMA_FLASH_ATTN_TYPE_DISABLED)
        context = api.llama_init_from_model(model, cp)
        if not context:
            raise RuntimeError("Pinned native extended context failed")
        data = run_workload(api, model, context, plan)
        artifact = store_logits(root, plan, data)
        return {
            "schema": SCHEMA, "status": "long_context_teacher_forced_complete",
            "model_version": "gemma-3-1b-it", "model_sha256": choice.expected_sha256,
            "quantization": "BF16", "gpu_uuid": uuid,
            "worker_pid": os.getpid(), "candidate": candidate,
            "n_ctx": plan.n_ctx, "n_batch": plan.n_batch,
            "n_ubatch": plan.n_ubatch, "swa_full": True,
            "flash_attention": plan.flash_attention,
            "type_k": plan.type_k, "type_v": plan.type_v,
            "prefill_tokens": plan.prompt_tokens, "decode_tokens": plan.decode_tokens,
            "warmups": WARMUPS, "repeats": REPEATS,
            "prompt_ids_sha256": hashlib.sha256(json.dumps(
                data["prompt"], separators=(",",":")).encode("ascii")).hexdigest(),
            "decode_ids": data["decode_ids"],
            "n_vocab": data["n_vocab"], "timings": data["timings"],
            "decode_logits_requested_per_token":
                data["decode_logits_requested_per_token"],
            "decode_logits_requested_count":
                data["decode_logits_requested_count"],
            "logits_artifact": artifact,
            "gpu_process_observed": bool(samples),
            "gpu_memory_peak_mib_sampled": max(samples) if samples else None,
            "nvtx_ranges_available": data["nvtx_ranges_available"],
            "inference_executed": True, "generated_tokens": 0,
            "production_qualified": False,
            "native_kernel_dispatch_verified": False,
        }
    finally:
        if context:
            api.llama_free(context)
        if model:
            api.llama_model_free(model)
        if initialized:
            api.llama_backend_free()
        event.set()
        gpuwatch.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 4:
        raise SystemExit("Expected root UUID candidate n_ctx")
    try:
        report = run_native(Path(args[0]), args[1], args[2], int(args[3]))
        print(PREFIX + json.dumps(report, sort_keys=True), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, TypeError, MemoryError, ImportError) as exc:
        print(PREFIX + json.dumps({
            "schema": SCHEMA, "status": "phase28_native_worker_failed",
            "error": type(exc).__name__ + ": " + str(exc)[:240],
            "inference_executed": None, "production_qualified": False,
            "native_kernel_dispatch_verified": False,
        }, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
