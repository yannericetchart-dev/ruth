"""Bounded native CUDA failure diagnostics for Ruth's isolated GGUF probe.

The caller supplies only its own llama.cpp/ggml native stderr. This is
evidence-based triage, NOT a guessed diagnosis from model size or Python
ValueError alone. Never emit arbitrary native stdout/stderr to JSON.
"""
from __future__ import annotations

import re

# Native upstream versions may use different text for the same failure.
_SIGNALS = (
    ("explicit_out_of_memory", re.compile(
        r"\bout.of.memory\b|cuda[_ ]error[_ ]out[_ ]of[_ ]memory|"
        r"cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED", re.I)),
    ("cuda_allocation_failed", re.compile(
        r"cudaMalloc(?:Async)?\s*(?:\(\s*\))?\s*(?:failed|error)|"
        r"failed to allocate\s+(?:CUDA|GPU|device)|"
        r"CUDA\d+\s+buffer\s+allocation\s+failed|"
        r"cuda allocation failed", re.I)),
    ("cuda_driver_or_runtime_error", re.compile(
        r"CUDA error|CUDA driver version|cudaGetDevice|"
        r"CUDA initialization failed|cudaErrorInvalidDevice", re.I)),
    ("gguf_or_tensor_unsupported", re.compile(
        r"unknown model architecture|unsupported (?:model|tensor|GGUF)|"
        r"invalid GGUF|unsupported.*(?:quantization|backend)", re.I)),
    ("model_load_failed_generic", re.compile(
        r"failed to load model|error loading model", re.I)),
)
_BUFFER_RE = re.compile(
    r"\bCUDA\d+\s+buffer size\s*=\s*(\d+(?:\.\d+)?)\s*MiB\b", re.I,
)
_PREFIX = re.compile(
    r"^(?:ggml_cuda|ggml_backend|llama_model|llama_load|"
    r"llama_context|gguf|CUDA|ggml-cuda|ggml_backend_cuda)", re.I,
)
_RELEVANT = re.compile(
    r"fail|error|alloc|out.of.memory|unsupported|"
    r"buffer size|memory (?:size|needed)|cudaMalloc|"
    r"invalid (?:model|tensor|GGUF)", re.I,
)


def _safe_native_lines(stderr: str) -> tuple[str, ...]:
    """Only native diagnostic prefixes; redact URLs and local filesystem paths."""
    items: list[str] = []
    for line in stderr.splitlines():
        line = line.strip()
        if not _PREFIX.match(line) or not _RELEVANT.search(line):
            continue
        line = re.sub(r"https?://\S+", "<url>", line)
        line = re.sub(r"(?:/home|/mnt|/tmp|/run/user)/\S+", "<path>", line)
        items.append(line[:220])
    return tuple(items[-8:])


def diagnose_cuda_failure(
    native_stderr: str, *, worker_error: str | None = None,
    returncode: int | None = None,
) -> dict:
    """Return only grounded categories, no invented GPU OOM claims."""
    source = native_stderr or ""
    detected = tuple(name for name, expr in _SIGNALS if expr.search(source))
    if "explicit_out_of_memory" in detected:
        category = "cuda_out_of_memory_reported"
    elif "cuda_allocation_failed" in detected:
        category = "cuda_allocation_failed_reported"
    elif "cuda_driver_or_runtime_error" in detected:
        category = "cuda_runtime_error_reported"
    elif "gguf_or_tensor_unsupported" in detected:
        category = "gguf_or_backend_unsupported_reported"
    elif "model_load_failed_generic" in detected:
        category = "native_model_load_failed_cause_unknown"
    else:
        category = "native_failure_cause_unknown"
    matches = [_BUFFER_RE.search(line) for line in source.splitlines()]
    mib = [float(m.group(1)) for m in matches if m is not None]
    return {
        "native_failure_category": category,
        "native_failure_signals": list(detected),
        "native_log_excerpt": list(_safe_native_lines(source)),
        "native_cuda_buffer_mib_reported": max(mib) if mib else None,
        "worker_exit_code": returncode,
        # A generic Python exception is not evidence for CUDA OOM.
        "generic_python_load_error_only": (
            category == "native_failure_cause_unknown"
            and bool(worker_error and "Failed to load model" in worker_error)
        ),
    }
