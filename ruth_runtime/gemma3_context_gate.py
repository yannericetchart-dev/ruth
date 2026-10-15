"""Fail-closed Gemma 3 1B CUDA context experiment gate.

The fixed 3x3x2 grid is an ALLOCATION-ONLY experiment, not evidence that
each dtype and attention mode can execute correctly. This module never
loads GGUF, touches CUDA, or approves inference or production.
"""
from __future__ import annotations

from typing import Any

GEMMA3_VERSION = "gemma-3-1b-it"
GEMMA3_KV_TYPES = ("F16", "BF16", "F32")
GEMMA3_FLASH_MODES = ("off", "on")
# Every ordered K/V combination is intentionally included for *both*
# explicit attention modes. AUTO and quantized KV remain excluded.
GEMMA3_CONTEXT_PAIRS = frozenset(
    (flash, k, v)
    for flash in GEMMA3_FLASH_MODES
    for k in GEMMA3_KV_TYPES
    for v in GEMMA3_KV_TYPES
)


def validate_gemma3_context_request(
    *, n_ctx: int, n_batch: int, n_ubatch: int,
    swa_full: bool, flash_attention: str,
    type_k: str, type_v: str, cuda_preinit: bool,
) -> dict[str, Any]:
    """Validate the exact 18-case context matrix; no automatic GPU execution."""
    if (type(n_ctx) is not int or n_ctx != 256
        or type(n_batch) is not int or n_batch != 64
        or type(n_ubatch) is not int or n_ubatch != 64
        or swa_full is not True or cuda_preinit is not False
        or (flash_attention, type_k, type_v) not in GEMMA3_CONTEXT_PAIRS):
        raise ValueError(
            "Gemma3 context probe only permits n_ctx=256, batch=ubatch=64, "
            "SWA FULL, no CUDA preinit, Flash ON/OFF and independent "
            "K/V storage in F16/BF16/F32 (18 combinations)"
        )
    return {
        "model": GEMMA3_VERSION, "experimental_allocation_only": True,
        "flash_attention": flash_attention, "type_k": type_k, "type_v": type_v,
        "inference_approved": False, "production_approved": False,
    }
