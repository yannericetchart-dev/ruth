"""Isolated, explicitly authorized Gemma3 GPU evaluator (NOT production).

Fixed public inputs; teacher-forced tokens, no sampling or open-ended generation.
Stores complete float32 logits locally to support actual numerical comparison.
Parent verifies wheel/GGUF/GPU, child repeats the frozen configuration checks.
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

from .gemma3_context_gate import validate_gemma3_context_request
from .inference_worker import measure_worker_gpu
from .load_only_worker import native_kv_dtype
from .gemma3_nvtx import NativeRanges
from .model_catalog import select_model
from .model_download import _local_path
from .install_plan import WHEEL_VERSION

SCHEMA = "ruth-gemma3-phase27-worker-v1"
PREFIX = "RUTH_GEMMA3_PHASE27_RESULT="
PROMPT = b"Ruth benchmarks deterministic attention precision."
CONTINUATION = b" One two three four five six seven eight nine ten."
DECODE_STEPS = 6
REPEATS = 3
WARMUPS = 1
N_CTX = 256
N_BATCH = 64
N_UBATCH = 64


def tokenizer_ids(api, vocab, text: bytes, *, special: bool) -> list[int]:
    tokens = (ctypes.c_int32 * N_BATCH)()
    count = api.llama_tokenize(
        vocab, text, len(text), tokens, N_BATCH, special, False)
    if type(count) is not int or not 1 <= count <= N_BATCH:
        raise RuntimeError("Pinned prompt tokenization exceeds deterministic batch")
    return [int(tokens[i]) for i in range(count)]


def evaluate_fixed_sequence(api, model, ctx) -> dict:
    import numpy as np

    vocab = api.llama_model_get_vocab(model)
    if not vocab:
        raise RuntimeError("Native Gemma3 vocabulary unavailable")
    n_vocab = api.llama_vocab_n_tokens(vocab)
    if type(n_vocab) is not int or not 512 <= n_vocab <= 300000:
        raise RuntimeError("Unverified native vocabulary size")
    prompt = tokenizer_ids(api, vocab, PROMPT, special=True)
    suffix = tokenizer_ids(api, vocab, CONTINUATION, special=False)
    if len(suffix) < DECODE_STEPS or len(prompt) + DECODE_STEPS >= N_CTX:
        raise RuntimeError("Unexpected pinned prefill/decode length")
    follow = suffix[:DECODE_STEPS]
    memory = api.llama_get_memory(ctx)
    if not memory:
        raise RuntimeError("Native KV memory unavailable")
    batch = api.llama_batch_init(N_BATCH, 0, 1)
    if not batch:
        raise RuntimeError("Native batch allocator failed")
    durations = []
    logits = []
    nvtx = NativeRanges()
    try:
        for trial in range(WARMUPS + REPEATS):
            api.llama_memory_clear(memory, True)
            prefill_ns = 0
            decode_ns = []
            rows = []
            stages = [(prompt, 0)] + [
                ([token], len(prompt) + i) for i, token in enumerate(follow)]
            for step, (tokens, start_pos) in enumerate(stages):
                batch.n_tokens = len(tokens)
                for i, token in enumerate(tokens):
                    batch.token[i] = token
                    batch.pos[i] = start_pos + i
                    batch.n_seq_id[i] = 1
                    batch.seq_id[i][0] = 0
                    batch.logits[i] = int(i == len(tokens) - 1)
                with nvtx.range("prefill" if step == 0 else "decode"):
                    t0 = monotonic_ns()
                    result = api.llama_decode(ctx, batch)
                    if result != 0:
                        raise RuntimeError(
                            f"llama_decode failed: phase={step}, code={result}")
                    api.llama_synchronize(ctx)
                    duration = monotonic_ns() - t0
                if duration <= 0:
                    raise RuntimeError("Invalid native decode elapsed time")
                if step == 0:
                    prefill_ns = duration
                else:
                    decode_ns.append(duration)
                if trial == WARMUPS + REPEATS - 1:
                    ptr = api.llama_get_logits_ith(ctx, -1)
                    if not ptr:
                        raise RuntimeError("Missing native logits")
                    # Pointer ownership remains native: copy before next decode.
                    values = np.ctypeslib.as_array(ptr, shape=(n_vocab,)).copy()
                    if values.dtype != np.dtype("float32") or not np.isfinite(values).all():
                        raise RuntimeError("Native logits nonfinite or wrong dtype")
                    rows.append(values)
            if trial >= WARMUPS:
                durations.append({
                    "prefill_ms": prefill_ns / 1_000_000.0,
                    "decode_ms_per_token": [x / 1_000_000.0 for x in decode_ns],
                })
            if rows:
                logits = rows
    finally:
        api.llama_batch_free(batch)
    if len(logits) != DECODE_STEPS + 1 or len(durations) != REPEATS:
        raise RuntimeError("Incomplete teacher-forced logits or repetition count")
    return {
        "logits": np.stack(logits, axis=0),
        "n_vocab": n_vocab,
        "prompt_ids": prompt,
        "decode_ids": follow,
        "timings": durations,
        "nvtx_ranges_available": nvtx.available,
    }


def write_local_logits(root: Path, payload: dict, flash: str, k: str, v: str) -> dict:
    import numpy as np
    base = root / ".ruth-build" / "phase27" / "gemma-3-1b-it"
    base.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in (base, base.parent, base.parent.parent)):
        raise RuntimeError("Refusing symlink benchmark output")
    # Unique, private, never overwritten; no Git publication or user prompts.
    fd, name = tempfile.mkstemp(
        prefix=f"flash-{flash}_k-{k}_v-{v}_",
        suffix=".npz", dir=base)
    path = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            np.savez_compressed(
                out, logits=payload["logits"],
                prompt_ids=np.asarray(payload["prompt_ids"], dtype=np.int32),
                decode_ids=np.asarray(payload["decode_ids"], dtype=np.int32))
        h = hashlib.sha256()
        with path.open("rb") as fp:
            for block in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(block)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {
        "relative_path": str(path.relative_to(root)),
        "sha256": h.hexdigest(),
        "logits_shape": list(payload["logits"].shape),
        "logits_dtype": "float32",
    }


def run_native(root: Path, uuid: str, flash: str, k: str, v: str) -> dict:
    validate_gemma3_context_request(
        n_ctx=N_CTX, n_batch=N_BATCH, n_ubatch=N_UBATCH,
        swa_full=True, flash_attention=flash,
        type_k=k, type_v=v, cuda_preinit=False)
    root = root.resolve(strict=True)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("GPU UUID binding mismatch")
    choice = select_model(
        "production", family="gemma", model="gemma-3",
        version="gemma-3-1b-it", format="gguf", quantization="BF16",
        project_root=root)
    file = _local_path(root, choice)
    if (file.is_symlink() or not file.is_file()
        or file.stat().st_size != choice.expected_size_bytes):
        raise RuntimeError("Pinned GGUF unavailable or altered")
    with file.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError("Invalid pinned GGUF magic")
    from llama_cpp import llama_cpp as api
    from llama_cpp import __version__ as binding_version
    if binding_version != WHEEL_VERSION or not api.llama_supports_gpu_offload():
        raise RuntimeError("Pinned native CUDA binding unavailable")
    stop = Event()
    samples = []
    watcher = Thread(target=measure_worker_gpu,
                     args=(os.getpid(), uuid, stop, samples), daemon=True)
    watcher.start()
    model = None
    ctx = None
    initialized = False
    try:
        api.llama_backend_init()
        initialized = True
        opts = api.llama_model_default_params()
        opts.n_gpu_layers = 2**31 - 1
        opts.main_gpu = 0
        opts.use_mmap = True
        model = api.llama_model_load_from_file(os.fsencode(file), opts)
        if not model:
            raise RuntimeError("Native GGUF model load failed")
        cfg = api.llama_context_default_params()
        cfg.n_ctx = N_CTX
        cfg.n_batch = N_BATCH
        cfg.n_ubatch = N_UBATCH
        cfg.n_seq_max = 1
        cfg.kv_unified = False
        cfg.swa_full = True
        cfg.offload_kqv = True
        cfg.type_k = native_kv_dtype(api, k)
        cfg.type_v = native_kv_dtype(api, v)
        cfg.flash_attn_type = (
            api.LLAMA_FLASH_ATTN_TYPE_ENABLED if flash == "on"
            else api.LLAMA_FLASH_ATTN_TYPE_DISABLED)
        ctx = api.llama_init_from_model(model, cfg)
        if not ctx:
            raise RuntimeError("Native Gemma3 context initialization failed")
        data = evaluate_fixed_sequence(api, model, ctx)
        artifact = write_local_logits(root, data, flash, k, v)
        return {
            "schema": SCHEMA, "status": "teacher_forced_gpu_workload_complete",
            "gpu_uuid": uuid, "worker_pid": os.getpid(),
            "model_sha256": choice.expected_sha256,
            "quantization": "BF16", "model_version": "gemma-3-1b-it",
            "flash_attention": flash, "type_k": k, "type_v": v,
            "n_ctx": N_CTX, "n_batch": N_BATCH, "n_ubatch": N_UBATCH,
            "warmups": WARMUPS, "repeats": REPEATS,
            "prefill_tokens": len(data["prompt_ids"]),
            "decode_tokens": DECODE_STEPS,
            "n_vocab": data["n_vocab"],
            "prompt_ids_sha256": hashlib.sha256(
                json.dumps(data["prompt_ids"], separators=(",", ":"))
                .encode("ascii")).hexdigest(),
            "decode_ids": data["decode_ids"],
            "artifact": artifact,
            "timings": data["timings"],
            "nvtx_ranges_available": data["nvtx_ranges_available"],
            "gpu_process_observed": bool(samples),
            "gpu_memory_peak_mib_observed": max(samples) if samples else None,
            "inference_executed": True,
            "generated_tokens": 0,  # Teacher-forcing, NO sampled generation.
            "teacher_forced_token_evaluations":
                (WARMUPS + REPEATS) * (len(data["prompt_ids"]) + DECODE_STEPS),
            "native_kernel_dispatch_verified": False,
            "numeric_fidelity_verified": False,
            "production_qualified": False,
        }
    finally:
        if ctx:
            api.llama_free(ctx)
        if model:
            api.llama_model_free(model)
        if initialized:
            api.llama_backend_free()
        stop.set()
        watcher.join(timeout=5)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 5:
        raise SystemExit("Expected project root, GPU UUID, Flash, type_k, type_v")
    try:
        report = run_native(Path(args[0]), *args[1:])
        print(PREFIX + json.dumps(report, sort_keys=True), flush=True)
        return 0
    except (OSError, RuntimeError, ValueError, ImportError, MemoryError) as exc:
        print(PREFIX + json.dumps({
            "schema": SCHEMA, "status": "phase27_native_worker_failed",
            "error": type(exc).__name__ + ": " + str(exc)[:240],
            "inference_executed": None,
            "native_kernel_dispatch_verified": False,
            "production_qualified": False,
        }, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
