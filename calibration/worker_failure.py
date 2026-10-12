"""Conservative interpretation of isolated llama.cpp CUDA worker failures.

A generic context-construction exception does NOT prove VRAM exhaustion.
Preserve bounded native stderr even when the Python worker exits cleanly,
because llama-cpp-python can report failed context initialization as a
ValueError with process returncode 0.
"""

import re


_OOM_EVIDENCE = (
    "out of memory",
    "cudamalloc failed",
    "cuda error 2",
)
_NATIVE_MARKERS = (
    "cuda", "ggml", "llama_context", "llama_new_context",
    "failed to allocate", "failed to create", "abort", "assert", "sigabrt",
    "memory", "allocation", "buffer",
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
    context_creation = (not success) and bool(re.search(
        r"failed\s+to\s+create\s+llama_context",
        worker_error + "\n" + stderr, flags=re.IGNORECASE,
    ))
    hints = [
        line.strip()[:400] for line in stderr.splitlines()
        if any(marker in line.lower() for marker in _NATIVE_MARKERS)
    ]
    classification = (
        "explicit_cuda_oom_evidence" if oom
        else "ambiguous_context_initialization" if context_creation
        else None
    )
    return {
        "suspected_cuda_oom": oom,
        "failure_hint": classification,
        "context_creation_failed": context_creation,
        "cuda_diagnostics": hints[-12:],
        # Critical: capture stderr for failed workers even with returncode=0.
        "stderr_tail": stderr[-2500:] if not success else None,
        "native_evidence_available": bool(stderr.strip()),
        "diagnostic_note": (
            "Un fallimento di llama_context non prova OOM CUDA. "
            "La classificazione usa solo indizi espliciti nei log nativi; "
            "SIGABRT o returncode=0 non bastano a determinarne la causa."
        ),
    }
