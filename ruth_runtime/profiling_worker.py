"""Bounded per-round CUDA profiling worker in Ruth's existing Python .venv.

Timing definitions:
 - model_load_seconds: backend initialization, GGUF/weights placement
 - prefill_seconds: llama.eval(all prompt token IDs), not a TTFT proxy
 - first_token_seconds: sample first token from prefill logits
 - decode_seconds: iterative llama.eval(previous token) + sample(next)
The warmup round is excluded. All input/model settings are fixed for a
comparable baseline, not representative of chat workloads.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from threading import Event, Thread
from time import perf_counter
from statistics import median

from .cuda_adapter import CudaLlamaAdapter, INITIAL_STACK
from .inference_worker import measure_worker_gpu
from .model_catalog import select_model
from .model_download import _local_path

PROFILE_PREFIX = "RUTH_GPU_PROFILE_RESULT="
PROMPT = (
    "Scrivi una lista numerata di venti semplici attività quotidiane, "
    "una per riga, con brevi spiegazioni pratiche in italiano. "
    "Continua la lista senza titoli o conclusioni."
)
SAMPLES = 5
WARMUP = 1
TOKENS = 64
CTX = 512
SEED = 42
PROFILE_FIELDS = (
    "tokenize_seconds", "prefill_seconds", "first_token_seconds",
    "decode_seconds", "prefill_tokens_per_second", "decode_tokens_per_second",
    "prompt_tokens", "completion_tokens", "decode_tokens",
)


def _stats(rows: list[dict]) -> dict:
    """No interpolation or invented missing throughput measurements."""
    result = {}
    for key in PROFILE_FIELDS:
        values = [row[key] for row in rows
                  if type(row.get(key)) in (float, int)
                  and row[key] >= 0]
        if len(values) != len(rows):
            result[key] = None
            continue
        result[key] = {
            "min": min(values), "median": median(values), "max": max(values),
        }
    return result


def profile_worker(root: Path, purpose: str, gpu_uuid: str) -> dict:
    root = root.resolve(strict=True)
    choice = select_model(purpose, project_root=root)
    if purpose != "test" or choice.version != "llama-3.2-1B-instruct" \
            or choice.quantization != "Q4_K_M":
        raise RuntimeError("Only pinned Llama 3.2 1B Q4_K_M test model supported")
    model = _local_path(root, choice)
    if model.is_symlink() or not model.is_file() \
            or model.stat().st_size != choice.expected_size_bytes:
        raise RuntimeError("Local GGUF missing or wrong size")
    with model.open("rb") as fp:
        if fp.read(4) != b"GGUF":
            raise RuntimeError("Not a GGUF model")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("Selected NVIDIA UUID not bound to worker")

    stop = Event()
    memory: list[int] = []
    watcher = Thread(
        target=measure_worker_gpu,
        args=(os.getpid(), gpu_uuid, stop, memory),
        daemon=True,
    )
    watcher.start()
    t0 = perf_counter()
    loaded = False
    rows: list[dict] = []
    try:
        adapter = CudaLlamaAdapter(native_diagnostics=True, seed=SEED)
        begin_load = perf_counter()
        session = adapter.load(
            model, INITIAL_STACK, {"n_ctx": CTX, "n_batch": 128, "n_ubatch": 128}
        )
        load_seconds = perf_counter() - begin_load
        loaded = True
        try:
            for index in range(WARMUP + SAMPLES):
                measured = session.benchmark_round(
                    PROMPT, max_tokens=TOKENS, seed=SEED,
                )
                if index >= WARMUP:
                    rows.append(measured)
        finally:
            session.close()
    finally:
        stop.set()
        watcher.join(timeout=8)

    return {
        "schema": "ruth-cuda-profile-v1",
        "purpose": purpose, "model_loaded": loaded,
        "gpu_uuid": gpu_uuid, "worker_pid": os.getpid(),
        "samples_requested": SAMPLES, "samples_completed": len(rows),
        "warmup_rounds": WARMUP, "n_ctx": CTX, "max_tokens": TOKENS,
        "seed": SEED, "n_gpu_layers_requested": -1, "cache_enabled": False,
        "model_load_seconds": load_seconds,
        "rounds": rows, "summary": _stats(rows),
        "gpu_process_observed": bool(memory),
        "gpu_memory_peak_mib_observed": max(memory) if memory else None,
        "worker_duration_seconds": perf_counter() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 3:
        raise SystemExit("Expected root, purpose, GPU UUID")
    try:
        value = profile_worker(Path(args[0]), args[1], args[2])
        print(PROFILE_PREFIX + json.dumps(value, sort_keys=True))
        success = (
            value["model_loaded"] and
            value["samples_completed"] == SAMPLES and
            all(row["decode_tokens"] >= 8 for row in value["rounds"])
        )
        return 0 if success else 2
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print(PROFILE_PREFIX + json.dumps({
            "schema": "ruth-cuda-profile-v1",
            "error": type(exc).__name__ + ": " + str(exc)[:180],
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
