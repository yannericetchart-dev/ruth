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


def kv_precision_candidates(profile: dict, *, n_ctx: int) -> dict:
    """Storage alternatives, not a performance or quality recommendation.

    BF16/F16 stored KV have exactly 2 bytes per scalar, F32 has 4.
    Attention projection WEIGHT dtype is a separate property from the
    precision of Key/Value ACTIVATIONS cached by llama.cpp.
    """
    geom = derived_gemma3_geometry(profile)
    obs = (profile["architecture"].get("gguf_header_observation") or {})
    coverage = obs.get("attention_projection_coverage")
    roles = ("attn_q", "attn_k", "attn_v", "attn_output")
    if (isinstance(coverage, dict) and
        all(isinstance(coverage.get(role), dict) and
            coverage[role].get("complete") is True for role in roles)):
        attn_formats = {
            role: coverage[role]["dtype_counts"] for role in roles
        }
        attn_status = "per_layer_attention_weight_formats_observed"
    else:
        attn_formats = None
        attn_status = "per_layer_attention_weight_formats_unknown"
    per_byte = n_ctx * (geom["global_layers"] + geom["swa_layers"]) * (
        geom["kv_heads"] * geom["key_head_dim"]
        + geom["kv_heads"] * geom["value_head_dim"])
    return {
        "status": "kv_precision_alternatives_source_derived_native_unverified",
        "weight_attention_dtypes_status": attn_status,
        "weight_attention_dtypes": attn_formats,
        "model_original_weight_format": "BF16",
        "kv_activations_are_not_weight_tensors": True,
        "native_context_dtype_support_verified": False,
        "automatic_cache_dtype_alignment_approved": False,
        "currently_approved_loading_baseline": {"type_k": "F16", "type_v": "F16"},
        "candidates": {
            dtype: {
                "type_k": dtype, "type_v": dtype,
                "bytes_per_scalar": n,
                "total_logical_mib_derived": round(per_byte*n/1048576, 4),
                "native_allocation_verified": False,
                "inference_numeric_fidelity_verified": False,
            } for dtype,n in (("F16",2),("BF16",2),("F32",4))
        },
        "evaluation_rule": (
            "The GGUF projection tensor storage formats inform provenance only. "
            "KV activation precision is independently tested for backend, "
            "CUDA allocation, token correctness and numeric stability."
        ),
    }


def flash_attention_kv_policy(*, flash_attention: str, type_k: str, type_v: str) -> dict:
    """Pinned-source comparison policy, never evidence of installed CUDA kernels.

    The pinned ggml/CMakeLists.txt defaults include f16-f16 and bf16-bf16
    FlashAttention vector template instances. fattn.cu can still route to
    MMA/TILE, where K and V are converted to f16, irrespective of cache dtype.
    """
    if flash_attention not in ("off", "on"):
        raise ValueError("Gemma 3 FlashAttention policy requires explicit on or off")
    if type_k not in ("F16", "BF16", "F32") or type_v not in (
            "F16", "BF16", "F32"):
        raise ValueError("Unqualified Gemma 3 KV precision")

    pair = (type_k, type_v)
    source_default_vec = pair in (("F16", "F16"), ("BF16", "BF16"))
    if flash_attention == "off":
        comparison = ("reference_baseline" if pair == ("F16", "F16")
                      else "experimental_unverified" if pair == ("BF16", "BF16")
                      else "diagnostic_only")
        return {
            "flash_attention": "off",
            "attention_path": "standard_non_flash",
            "comparison_tier": comparison,
            "flash_vector_pair_in_pinned_cmake_defaults": None,
            "flash_f16_conversion_possible": None,
            "installed_cuda_kernel_variants_verified": False,
            "runtime_kernel_dispatch_verified": False,
            "native_context_approved": False,
            "source_note": (
                "The pinned llama-context.cpp requires Flash Attention "
                "for quantized V, not for non-quantized F16/BF16/F32 V. "
                "Native non-Flash dtype and numerical support remain unverified."
            ),
        }

    comparison = ("reference_candidate" if pair == ("F16", "F16")
                  else "source_default_vector_candidate"
                  if pair == ("BF16", "BF16") else "conversion_risk_diagnostic")
    return {
        "flash_attention": "on",
        "attention_path": "flash_dispatch_runtime_unverified",
        "comparison_tier": comparison,
        "flash_vector_pair_in_pinned_cmake_defaults": source_default_vec,
        "flash_f16_conversion_possible": True,
        "installed_cuda_kernel_variants_verified": False,
        "runtime_kernel_dispatch_verified": False,
        "native_context_approved": False,
        "source_note": (
            "Pinned fattn.cu may dispatch to VEC, MMA_F16 or TILE. "
            "MMA/TILE use temporary f16 K/V; the vector path accepts "
            "matching source-default pairs, but f32 or missing vector "
            "instances can require temporary f16 conversion. "
            "CMake source defaults do not establish the installed binary."
        ),
    }


def calculate_gemma3_kv(*, project_root: Path = PROJECT_ROOT, n_ctx: int = 256,
                         n_batch: int = 64, n_ubatch: int = 64,
                         swa_full: bool = True, flash_attention: str = "off",
                         type_k: str = "F16", type_v: str = "F16") -> dict:
    """Strict load-phase logical tensor forecast, NO native alloc/GPU operations."""
    # Read-only STORAGE estimation, not permission for a native context.
    bytes_per_scalar = {"F16": 2, "BF16": 2, "F32": 4}
    if (type(n_ctx) is not int or n_ctx < 256 or n_ctx > 32768 or n_ctx % 256 != 0
        or type(n_batch) is not int or type(n_ubatch) is not int
        or not 1 <= n_ubatch <= n_batch <= n_ctx
        or swa_full is not True or flash_attention not in ("off", "on")
        or type_k not in bytes_per_scalar or type_v not in bytes_per_scalar):
        raise ValueError(
            "Gemma 3 KV plan requires 256-step aligned context, "
            "F16/BF16/F32, explicit Flash ON/OFF, SWA FULL")
    profile = load_profile("gemma","gemma-3","gemma-3-1b-it", project_root=project_root)
    geom = derived_gemma3_geometry(profile)
    choices = kv_precision_candidates(profile,n_ctx=n_ctx)
    flash_policy = flash_attention_kv_policy(
        flash_attention=flash_attention, type_k=type_k, type_v=type_v)
    # n_seq_max 1, n_ctx padded to 256. For swa_full=true and context
    # smaller/larger than the SWA window, allocation has n_ctx cells
    # per SWA layer; this number is logical, NOT CUDA buffer size.
    cells = n_ctx
    key_bytes_per_layer = (cells * geom["kv_heads"] * geom["key_head_dim"]
                           * bytes_per_scalar[type_k])
    value_bytes_per_layer = (cells * geom["kv_heads"] * geom["value_head_dim"]
                             * bytes_per_scalar[type_v])
    combined_bytes_per_layer = key_bytes_per_layer + value_bytes_per_layer
    global_bytes = geom["global_layers"] * combined_bytes_per_layer
    swa_bytes = geom["swa_layers"] * combined_bytes_per_layer
    key_bytes = (geom["global_layers"] + geom["swa_layers"]) * key_bytes_per_layer
    value_bytes = (geom["global_layers"] + geom["swa_layers"]) * value_bytes_per_layer
    return {
        "status": "gemma3_kv_pinned_source_derived_native_context_unverified",
        "model": "gemma-3-1b-it", "variant": "BF16",
        "llama_cpp_commit": BACKEND_REVISION,
        "source_derived_geometry": geom,
        "kv_precision_candidates": choices,
        "kv_attention_policy": flash_policy,
        "configuration": {
            "n_ctx_requested": n_ctx, "n_ctx_effective_derived": cells,
            "n_batch": n_batch, "n_ubatch": n_ubatch,
            "type_k": type_k, "type_v": type_v,
            "swa_full": swa_full, "flash_attention": flash_attention,
            "n_seq_max": 1, "kv_unified": False,
        },
        "selected_precision": {
            "is_current_loading_baseline": (
                flash_attention == "off" and type_k == "F16" and type_v == "F16"),
            "native_context_dtype_support_verified": False,
            "native_allocation_verified": False,
            "automatic_alignment_approved": False,
            "planning_only": True,
        },
        "logical_kv": {
            "global_layers": geom["global_layers"],
            "swa_layers": geom["swa_layers"],
            "cells_per_layer": cells,
            "key_logical_mib_derived": round(key_bytes/1048576, 4),
            "value_logical_mib_derived": round(value_bytes/1048576, 4),
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
