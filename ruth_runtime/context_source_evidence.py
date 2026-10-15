"""Source-only, model-specific context memory evidence for pinned native probes.

Preserve the existing public Gemma2 contract. Gemma3's storage-only plan
is normalized to the same archival keys; unknown SWA-off thresholds stay None.
No CUDA allocation, Flash kernel execution or numeric-fidelity inference.
"""
from __future__ import annotations

from pathlib import Path

from .dependencies import PROJECT_ROOT
from .gemma3_context_gate import GEMMA3_VERSION, validate_gemma3_context_request
from .gemma3_kv_profile import calculate_gemma3_kv
from .llama_kv_profile import calculate_llama_kv


def calculate_context_source(
    *, model_version: str, project_root: Path = PROJECT_ROOT,
    n_ctx: int, n_batch: int, n_ubatch: int,
    swa_full: bool, flash_attention: str,
    type_k: str, type_v: str,
) -> dict:
    if model_version != GEMMA3_VERSION:
        if model_version != "gemma-2-9B-it":
            raise ValueError("Unqualified pinned native context model")
        return calculate_llama_kv(
            project_root=project_root, n_ctx=n_ctx, n_batch=n_batch,
            n_ubatch=n_ubatch, swa_full=swa_full,
            flash_attention=flash_attention, type_k=type_k, type_v=type_v,
        )
    validate_gemma3_context_request(
        n_ctx=n_ctx, n_batch=n_batch, n_ubatch=n_ubatch,
        swa_full=swa_full, flash_attention=flash_attention,
        type_k=type_k, type_v=type_v, cuda_preinit=False,
    )
    planned = calculate_gemma3_kv(
        project_root=project_root, n_ctx=n_ctx, n_batch=n_batch,
        n_ubatch=n_ubatch, swa_full=swa_full,
        flash_attention=flash_attention, type_k=type_k, type_v=type_v,
    )
    geom = planned["source_derived_geometry"]
    cells = planned["logical_kv"]["cells_per_layer"]
    scalar_bytes = {"F16": 2, "BF16": 2}
    key_layer_bytes = cells * geom["kv_heads"] * geom["key_head_dim"] * scalar_bytes[type_k]
    value_layer_bytes = cells * geom["kv_heads"] * geom["value_head_dim"] * scalar_bytes[type_v]
    return {
        "configuration": {
            "n_ctx_effective_derived": cells,
            "n_ubatch_effective_derived": n_ubatch,
            "n_batch_requested": n_batch,
            "n_ubatch_requested": n_ubatch,
            "swa_full": swa_full,
            "flash_attention_requested": flash_attention,
            "type_k": type_k, "type_v": type_v,
        },
        "global_cache": {"cells": cells},
        "swa_cache": {"cells": cells},
        "key_logical_bytes_derived": (geom["global_layers"] + geom["swa_layers"]) * key_layer_bytes,
        "value_logical_bytes_derived": (geom["global_layers"] + geom["swa_layers"]) * value_layer_bytes,
        "swa_full_off_saves_bytes_derived": None,
        "swa_off_first_saving_at_effective_ctx": None,
        "model_geometry_verification_pending": True,
        "native_allocation_executed": False,
    }


def context_source_derivation(source: dict) -> dict:
    cfg = source["configuration"]
    return {
        "n_ctx_effective_derived": cfg["n_ctx_effective_derived"],
        "n_ubatch_effective_derived": cfg["n_ubatch_effective_derived"],
        "global_cells_derived": source["global_cache"]["cells"],
        "swa_cells_derived": source["swa_cache"]["cells"],
        "key_logical_bytes_derived": source["key_logical_bytes_derived"],
        "value_logical_bytes_derived": source["value_logical_bytes_derived"],
        "swa_full_off_saves_bytes_derived": source["swa_full_off_saves_bytes_derived"],
        "swa_off_first_saving_at_effective_ctx":
            source["swa_off_first_saving_at_effective_ctx"],
        "model_geometry_verification_pending": True,
    }
