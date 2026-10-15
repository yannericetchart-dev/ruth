"""Source-derived Gemma 3 1B IT KV geometry from its pinned GGUF header.

Only 26-layer/1-KV-head pinned BF16 GGUF is accepted. This is NOT a
CUDA allocation test or proof that a given context initializes.
The SWA/default-pattern semantics come from the pinned llama.cpp source:
src/models/gemma3.cpp load_swa_pattern(ml, 6)
src/llama-hparams.cpp set_swa_pattern(..., dense_first=false).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dependencies import PROJECT_ROOT
from .model_profile_v2 import load_profile

BACKEND_REVISION = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"
GGUF_SHA256 = "b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf"
SOURCE_PATHS = (
    "src/models/gemma3.cpp",
    "src/llama-hparams.cpp",
    "src/llama-context.cpp",
)

def derived_gemma3_geometry(profile: dict) -> dict[str, Any]:
    if ((profile.get("model") or {}).get("version") != "gemma-3-1b-it"
        or (profile.get("variants") or {}).get("BF16", {}).get("gguf_sha256") != GGUF_SHA256
        or ((profile.get("backend_policies") or {}).get("llama.cpp") or {}).get(
            "pinned_revision") != BACKEND_REVISION):
        raise ValueError("Unpinned Gemma 3 source or backend revision")

    arch = profile.get("architecture") or {}
    obs = arch.get("gguf_header_observation") or {}
    if (obs.get("status") != "gguf_header_observed_native_crosscheck_pending"
        or obs.get("gguf_sha256") != GGUF_SHA256):
        raise ValueError("Pinned GGUF geometry not yet observed")
    meta = obs.get("metadata_fields") or {}
    required = {
        "general.architecture": "gemma3",
        "gemma3.block_count": 26,
        "gemma3.attention.head_count": 4,
        "gemma3.attention.head_count_kv": 1,
        "gemma3.attention.key_length": 256,
        "gemma3.attention.value_length": 256,
        "gemma3.attention.sliding_window": 512,
    }
    if any(meta.get(k) != v for k,v in required.items()):
        raise ValueError("Unsupported or incomplete native Gemma 3 GGUF KV geometry")
    # The source's fallback is applied only if no explicit per-layer
    # sliding-window pattern key is present.
    if "gemma3.attention.sliding_window_pattern" in meta:
        pattern = meta["gemma3.attention.sliding_window_pattern"]
        if (not isinstance(pattern, list) or len(pattern) != 26
            or any(type(v) not in (int, bool) or v not in (0,1)
                   for v in pattern)):
            raise ValueError("Unknown explicit Gemma 3 sliding-window pattern")
        swa = [i for i,v in enumerate(pattern) if v]
        global_ = [i for i,v in enumerate(pattern) if not v]
        source = "explicit_gguf_layer_array"
    else:
        # Gemma3 defaults to five SWA layers then one global (6-cycle).
        swa = [i for i in range(26) if i % 6 < 5]
        global_ = [i for i in range(26) if i % 6 == 5]
        source = "pinned_llama_cpp_default_6_dense_last"
    if not swa or not global_ or len(swa)+len(global_) != 26:
        raise ValueError("Unsupported Gemma 3 attention partition")
    return {
        "status": "pinned_source_derived_native_context_unverified",
        "gguf_sha256": GGUF_SHA256,
        "llama_cpp_commit": BACKEND_REVISION,
        "llama_cpp_source_paths": list(SOURCE_PATHS),
        "swa_pattern_source": source,
        "global_layer_indices": global_,
        "swa_layer_indices": swa,
        "global_layers": len(global_),
        "swa_layers": len(swa),
        "swa_window": 512,
        "kv_heads": 1,
        "key_head_dim": 256,
        "value_head_dim": 256,
        "context_alignment": 256,
    }


def calculate_gemma3_kv(*, project_root: Path = PROJECT_ROOT, n_ctx: int = 256,
                         n_batch: int = 64, n_ubatch: int = 64,
                         swa_full: bool = True, flash_attention: str = "off",
                         type_k: str = "F16", type_v: str = "F16") -> dict:
    """Strict load-phase logical tensor forecast, NO native alloc/GPU operations."""
    if (type(n_ctx) is not int or n_ctx < 256 or n_ctx > 32768 or n_ctx % 256 != 0
        or type(n_batch) is not int or type(n_ubatch) is not int
        or not 1 <= n_ubatch <= n_batch <= n_ctx
        or (swa_full, flash_attention, type_k, type_v) != (True,"off","F16","F16")):
        raise ValueError("Gemma 3 load-phase KV fixed to 256-step context, F16/F16, Flash OFF, SWA FULL")
    profile = load_profile("gemma","gemma-3","gemma-3-1b-it", project_root=project_root)
    geom = derived_gemma3_geometry(profile)
    # n_seq_max 1, n_ctx padded to 256. For swa_full=true and context
    # smaller/larger than the SWA window, allocation has n_ctx cells
    # per SWA layer; this number is logical, NOT CUDA buffer size.
    cells = n_ctx
    per_layer = (cells * geom["kv_heads"]
                 * (geom["key_head_dim"] * 2 + geom["value_head_dim"] * 2))
    global_bytes = geom["global_layers"] * per_layer
    swa_bytes = geom["swa_layers"] * per_layer
    return {
        "status": "gemma3_kv_pinned_source_derived_native_context_unverified",
        "model": "gemma-3-1b-it", "variant": "BF16",
        "llama_cpp_commit": BACKEND_REVISION,
        "source_derived_geometry": geom,
        "configuration": {
            "n_ctx_requested": n_ctx, "n_ctx_effective_derived": cells,
            "n_batch": n_batch, "n_ubatch": n_ubatch,
            "type_k": type_k, "type_v": type_v,
            "swa_full": swa_full, "flash_attention": flash_attention,
            "n_seq_max": 1, "kv_unified": False,
        },
        "logical_kv": {
            "global_layers": geom["global_layers"],
            "swa_layers": geom["swa_layers"],
            "cells_per_layer": cells,
            "key_logical_mib_derived": round(
                (global_bytes + swa_bytes)/2/1048576, 4),
            "value_logical_mib_derived": round(
                (global_bytes + swa_bytes)/2/1048576, 4),
            "global_logical_mib_derived": round(global_bytes/1048576, 4),
            "swa_logical_mib_derived": round(swa_bytes/1048576, 4),
            "total_logical_mib_derived": round(
                (global_bytes+swa_bytes)/1048576,4),
        },
        "native_allocation_executed": False,
        "gpu_fit_verified": False,
        "inference_executed": False,
        "context_loading_approved": False,
    }
