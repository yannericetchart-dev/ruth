"""Fail-closed Gemma 3 1B CUDA context experiment gate.

These two pairs are the ONLY currently staged probes. This module does not
load GGUF, touch CUDA, allocate context memory or qualify production.
"""
from __future__ import annotations

from typing import Any

GEMMA3_VERSION = "gemma-3-1b-it"
GEMMA3_CONTEXT_PAIRS = frozenset({
    ("off", "F16", "F16"),
    ("on", "BF16", "BF16"),
})


def validate_gemma3_context_request(
    *, n_ctx: int, n_batch: int, n_ubatch: int,
    swa_full: bool, flash_attention: str,
    type_k: str, type_v: str, cuda_preinit: bool,
) -> dict[str, Any]:
    """A narrow experimental allocation-only allowlist, not inference permission."""
    if (type(n_ctx) is not int or n_ctx != 256
        or type(n_batch) is not int or n_batch != 64
        or type(n_ubatch) is not int or n_ubatch != 64
        or swa_full is not True or cuda_preinit is not False
        or (flash_attention, type_k, type_v) not in GEMMA3_CONTEXT_PAIRS):
        raise ValueError(
            "Gemma3 context probe only permits n_ctx=256, batch=ubatch=64, "
            "SWA FULL, no CUDA preinit, and F16/F16 Flash OFF or BF16/BF16 Flash ON"
        )
    return {
        "model": GEMMA3_VERSION, "experimental_allocation_only": True,
        "flash_attention": flash_attention, "type_k": type_k, "type_v": type_v,
        "inference_approved": False, "production_approved": False,
    }
