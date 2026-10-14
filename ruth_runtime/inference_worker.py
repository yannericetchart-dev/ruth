"""Isolated single-model CUDA smoke worker. Never import this in launcher.

Input from Ruth's own previously verified catalog (not an arbitrary model path).
Machine-verifiable output is JSON, with no model content or personal prompts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import monotonic

from .cuda_adapter import CudaLlamaAdapter, INITIAL_STACK
from .model_catalog import select_model
from .model_download import _local_path


def _sample_compute_pid(pid: int, gpu_uuid: str) -> int | None:
    """Return this worker's GPU allocated MiB (None means unobserved)."""
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if probe.returncode:
            return None
        for line in probe.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            if parts[0] == str(pid) and parts[1] == gpu_uuid:
                val = parts[2]
                if val.isdecimal() and int(val) > 0:
                    return int(val)
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def measure_worker_gpu(pid: int, uuid: str, stop: Event, values: list[int]) -> None:
    while not stop.is_set():
        used = _sample_compute_pid(pid, uuid)
        if used is not None:
            values.append(used)
        stop.wait(0.2)


def run_worker(project_root: Path, purpose: str, gpu_uuid: str) -> dict:
    # The parent has already checked the whole GGUF digest and CPU/GPU plan.
    # Worker still checks path, manifest and magic; it must not trust arbitrary
    # filenames from argv. No network, no cache, no CPU fallback.
    root = Path(project_root).resolve(strict=True)
    choice = select_model(purpose, project_root=root)
    if purpose != "test" or choice.version != "llama-3.2-1B-instruct":
        raise RuntimeError("Only the pinned small GGUF smoke model is allowed")
    file = _local_path(root, choice)
    if file.is_symlink() or not file.is_file():
        raise RuntimeError("Pinned local GGUF not found")
    if file.stat().st_size != choice.expected_size_bytes:
        raise RuntimeError("Local GGUF size changed since parent preflight")
    with file.open("rb") as fp:
        if fp.read(4) != b"GGUF":
            raise RuntimeError("Local model is not GGUF")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("Worker GPU binding did not match selected GPU UUID")

    stop = Event()
    memory_mib: list[int] = []
    watcher = Thread(target=measure_worker_gpu,
                     args=(os.getpid(), gpu_uuid, stop, memory_mib),
                     daemon=True)
    t0 = monotonic()
    watcher.start()
    loaded = False
    tokens = 0
    text_hash = None
    try:
        engine = CudaLlamaAdapter(native_diagnostics=True)
        session = engine.load(
            file, INITIAL_STACK, {"n_ctx": 512, "n_batch": 128, "n_ubatch": 128}
        )
        loaded = True
        try:
            completion = session.generate_report(
                "Rispondi in italiano con una sola parola: ciao",
                max_tokens=24,
            )
            tokens = completion["completion_tokens"]
            text_hash = hashlib.sha256(
                completion["text"].encode("utf-8")).hexdigest()
        finally:
            session.close()
    finally:
        stop.set()
        watcher.join(timeout=8)
    return {
        "schema": "ruth-cuda-gguf-smoke-v1",
        "purpose": purpose, "model_loaded": loaded,
        "completion_tokens": tokens, "generated_text_sha256": text_hash,
        "gpu_uuid": gpu_uuid, "worker_pid": os.getpid(),
        "gpu_process_observed": bool(memory_mib),
        "gpu_memory_peak_mib_observed": max(memory_mib) if memory_mib else None,
        "worker_duration_seconds": round(monotonic() - t0, 3),
        "cache_enabled": False,
    }


def main(args: list[str] | None = None) -> int:
    argv = args if args is not None else sys.argv[1:]
    if len(argv) != 3:
        raise SystemExit("Worker requires root, test purpose and selected GPU UUID")
    try:
        report = run_worker(Path(argv[0]), argv[1], argv[2])
        print("RUTH_GPU_WORKER_RESULT=" + json.dumps(report, sort_keys=True))
        return 0 if report["model_loaded"] and report["completion_tokens"] > 0 else 2
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        print("RUTH_GPU_WORKER_RESULT=" + json.dumps({
            "schema": "ruth-cuda-gguf-smoke-v1", "model_loaded": False,
            "completion_tokens": 0, "error": type(exc).__name__ +
            ": " + str(exc)[:180],
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
