"""Isolated CUDA-only Gemma production *candidate* minimum-context worker.

Not a production runtime. The parent verifies the complete pinned local GGUF
SHA256 and license manifest before starting this process. No CPU fallback,
network, install, quantization auto-switch, or context escalation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import monotonic
import sys

from .cuda_adapter import CudaLlamaAdapter, INITIAL_STACK
from .inference_worker import measure_worker_gpu
from .model_catalog import select_model
from .model_download import _local_path

SCHEMA = "ruth-gemma-min-context-v1"
PREFIX = "RUTH_GEMMA_MIN_CONTEXT_RESULT="
N_CTX = 128
N_BATCH = 64
N_UBATCH = 64
MAX_TOKENS = 24
SEED = 42
PROMPT = "Rispondi in italiano con una sola parola: ciao"


def run_production_worker(root: Path, gpu_uuid: str) -> dict:
    root = root.resolve(strict=True)
    choice = select_model("production", project_root=root)
    if (choice.family, choice.model, choice.version, choice.quantization) != (
        "gemma", "gemma-2", "gemma-2-9B-it", "Q4_K_L"
    ):
        raise RuntimeError("Only pinned Gemma 2 9B IT Q4_K_L allowed")
    file = _local_path(root, choice)
    if file.is_symlink() or not file.is_file() or \
            file.stat().st_size != choice.expected_size_bytes:
        raise RuntimeError("Pinned Gemma GGUF missing or wrong size")
    with file.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError("Invalid GGUF magic")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("Worker GPU UUID binding mismatch")
    samples: list[int] = []
    event = Event()
    watcher = Thread(
        target=measure_worker_gpu,
        args=(os.getpid(), gpu_uuid, event, samples), daemon=True,
    )
    start = monotonic()
    watcher.start()
    model_loaded = False
    generated = 0
    result_hash = None
    try:
        adapter = CudaLlamaAdapter(native_diagnostics=True, seed=SEED)
        session = adapter.load(
            file, INITIAL_STACK,
            {"n_ctx": N_CTX, "n_batch": N_BATCH, "n_ubatch": N_UBATCH},
        )
        model_loaded = True
        try:
            result = session.generate_report(PROMPT, max_tokens=MAX_TOKENS)
            generated = result["completion_tokens"]
            result_hash = hashlib.sha256(
                result["text"].encode("utf-8")).hexdigest()
        finally:
            session.close()
    finally:
        event.set()
        watcher.join(timeout=8)
    return {
        "schema": SCHEMA, "purpose": "production",
        "model_loaded": model_loaded,
        "generated_text_sha256": result_hash,
        "completion_tokens": generated, "n_ctx": N_CTX,
        "n_batch": N_BATCH, "n_ubatch": N_UBATCH,
        "max_tokens": MAX_TOKENS, "seed": SEED,
        "gpu_uuid": gpu_uuid, "worker_pid": os.getpid(),
        "gpu_process_observed": bool(samples),
        "gpu_memory_peak_mib_observed": max(samples) if samples else None,
        "worker_duration_seconds": round(monotonic() - start, 3),
        "cpu_fallback_selected": False, "model_cache_enabled": False,
    }


def main(args: list[str] | None = None) -> int:
    argv = args if args is not None else sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit("Expected project root and selected GPU UUID")
    try:
        result = run_production_worker(Path(argv[0]), argv[1])
        print(PREFIX + json.dumps(result, sort_keys=True))
        return 0 if result["model_loaded"] and result["completion_tokens"] > 0 else 2
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(PREFIX + json.dumps({
            "schema": SCHEMA,
            "error": type(exc).__name__ + ": " + str(exc)[:200],
            "model_loaded": False, "completion_tokens": 0,
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
