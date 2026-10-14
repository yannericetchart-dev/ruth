"""Conservative interpretation of isolated llama.cpp CUDA worker failures.

A generic context-construction exception does NOT prove VRAM exhaustion.
Preserve bounded native stderr even when the Python worker exits cleanly,
because llama-cpp-python can report failed context initialization as a
ValueError with process returncode 0.
"""

import re
import signal


_OOM_EVIDENCE = (
    "out of memory",
    "cudamalloc failed",
    "cuda error 2",
)
# cuBLAS handle creation can explicitly fail to allocate GPU resources
# without a cudaMalloc OOM line. This is a *runtime feasibility limit*,
# not proof that a specific number of bytes of VRAM was exhausted.
_CUDA_RESOURCE_FAILURE = re.compile(
    r"(?:cuda error:\s*the resource allocation failed|"
    r"cublas_status_alloc_failed)",
    flags=re.IGNORECASE,
)
_NATIVE_MARKERS = (
    "cuda", "ggml", "llama_context", "llama_new_context",
    "failed to allocate", "failed to create", "abort", "assert", "sigabrt",
    "memory", "allocation", "buffer", "cublas", "cudnn", "fatal", "error",
)


def analyze_worker_failure(stderr: str, error: str | None,
                           returncode: int, success: bool) -> dict:
    if not isinstance(stderr, str) or not isinstance(returncode, int):
        raise ValueError("stderr e codice di ritorno del worker non validi.")
    native_lower = stderr.lower()
    worker_error = (error or "").strip()
    if not isinstance(worker_error, str):
        raise ValueError("Errore del worker non valido.")
    oom = (not success) and any(marker in native_lower for marker in _OOM_EVIDENCE)
    allocation_failed = (
        not success and not oom
        and bool(_CUDA_RESOURCE_FAILURE.search(stderr))
    )
    context_creation = (not success) and bool(re.search(
        r"failed\s+to\s+create\s+llama_context",
        worker_error + "\n" + stderr, flags=re.IGNORECASE,
    ))
    hints = [
        line.strip()[:400] for line in stderr.splitlines()
        if any(marker in line.lower() for marker in _NATIVE_MARKERS)
    ]
    phases = re.findall(
        r"^RUTH_WORKER_PHASE=([a-z_]+)\s*$",
        stderr, flags=re.MULTILINE,
    )
    last_worker_phase = phases[-1] if phases else None
    # The last flushed phase marker is evidence of where the worker got.
    # It is NOT a diagnosis of why CUDA aborted.
    failed_during = (
        "load" if last_worker_phase == "load_begin"
        else "prefill" if last_worker_phase == "prefill_begin"
        else "decode" if last_worker_phase == "decode_begin"
        else "after_decode" if last_worker_phase == "decode_complete"
        else "between_phases" if last_worker_phase in (
            "load_complete", "prefill_complete",
        ) else "unknown"
    )
    error_lines = [
        line.strip()[:600] for line in stderr.splitlines()
        if re.search(
            r"(?:CUDA error|cublas_status|cublas_handle|"
            r"cudaMalloc failed|resource allocation|"
            r"out of memory|assert|fatal|abort|failed to alloc)",
            line, flags=re.IGNORECASE,
        )
    ]
    classification = (
        "explicit_cuda_oom_evidence" if oom
        else "explicit_cuda_resource_allocation_failure" if allocation_failed
        else "ambiguous_context_initialization" if context_creation
        else None
    )
    return {
        "suspected_cuda_oom": oom,
        "native_cuda_resource_allocation_failed": allocation_failed,
        "failure_hint": classification,
        "context_creation_failed": context_creation,
        "cuda_diagnostics": hints[-24:],
        "fatal_native_lines": error_lines[-16:],
        "abort_signal_detected": returncode == -signal.SIGABRT,
        "last_worker_phase": last_worker_phase,
        "failed_during": failed_during if not success else None,
        # Critical: capture stderr for failed workers even with returncode=0.
        "stderr_tail": stderr[-6000:] if not success else None,
        "native_evidence_available": bool(stderr.strip()),
        "diagnostic_note": (
            "Un fallimento di llama_context non prova OOM CUDA. "
            "La classificazione usa solo indizi espliciti nei log nativi; "
            "SIGABRT o returncode=0 non bastano a determinarne la causa. "
            "Un errore di allocazione cuBLAS limita il runtime, ma non "
            "dimostra quanti byte di VRAM siano esauriti."
        ),
    }
