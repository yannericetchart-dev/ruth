"""Exact-version llama.cpp KV capacity planner. All results are DERIVED, never measured.

Pinned to upstream 0c1e57098bba43ac29e6e3b677cdceebdd22334f.
Does not import llama_cpp, touch the GPU, perform I/O other than policy, or
accept unqualified gguf metadata as native measurements.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .dependencies import PROJECT_ROOT
from .install_plan import LLAMA_CPP_SUBMODULE_COMMIT, WHEEL_VERSION

POLICY_FILE = "catalog/llama_kv_backend_policy.json"
SUPPORTED_KV_TYPES = ("F16", "Q8_0", "Q4_0")
SUPPORTED_FLASH = ("off", "on", "auto")


def load_llama_kv_policy(project_root: Path = PROJECT_ROOT) -> dict:
    path = Path(project_root) / POLICY_FILE
    policy = json.loads(path.read_text(encoding="utf-8"))
    if (policy.get("schema") != "ruth-llama-kv-backend-policy-v1"
        or policy.get("backend") != "llama.cpp"
        or policy.get("upstream_commit") != LLAMA_CPP_SUBMODULE_COMMIT
        or policy.get("binding_version") != WHEEL_VERSION
        or policy.get("context_alignment_cells") != 256
        or policy.get("swa_alignment_cells") != 256
        or policy.get("graph_min_alignment_cells") != 256):
        raise ValueError("Unverified or modified llama.cpp KV backend policy")
    return policy


def pad(value: int, alignment: int) -> int:
    if type(value) is not int or value < 1 or type(alignment) is not int or alignment < 1:
        raise ValueError("Positive integer pad inputs required")
    return (value + alignment - 1) // alignment * alignment


def _bytes_for_rows(cells: int, elements: int, dtype: str, types: dict) -> int:
    item = types[dtype]
    block, bytes_per = item["block_elements"], item["bytes_per_block"]
    if elements % block:
        raise ValueError("K/V head dimension incompatible with quantization block")
    return cells * (elements // block) * bytes_per


def calculate_llama_kv(
    *, model_version: str = "gemma-2-9B-it",
    n_ctx: int, n_batch: int, n_ubatch: int,
    swa_full: bool, flash_attention: str = "off",
    type_k: str = "F16", type_v: str = "F16",
    n_seq_max: int = 1, kv_unified: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Source-based memory estimate for this EXACT backend source and model default.

    No assumption of actual tensor allocation, backend buffer padding or GPU fit.
    Require single sequence for the current Gemma production worker; other
    sequence modes need their own profiler and are not inferred as validated.
    """
    p = load_llama_kv_policy(project_root)
    if model_version not in p["model_defaults"]:
        raise ValueError("Model architecture has no audited KV geometry")
    if (type(n_ctx) is not int or not 1 <= n_ctx <= 131072
        or type(n_batch) is not int or not 1 <= n_batch <= 131072
        or type(n_ubatch) is not int or not 1 <= n_ubatch <= 131072
        or type(n_seq_max) is not int or n_seq_max != 1
        or kv_unified is not False
        or type(swa_full) is not bool):
        raise ValueError("Only positive, single-sequence Gemma2 KV profiles supported")
    if flash_attention not in SUPPORTED_FLASH:
        raise ValueError("Flash Attention must be off/on/auto")
    if type_k not in SUPPORTED_KV_TYPES or type_v not in SUPPORTED_KV_TYPES:
        raise ValueError("Unqualified KV tensor precision")
    if flash_attention == "off" and type_v != "F16":
        raise ValueError("Quantized V requires enabled Flash Attention")
    g = p["model_defaults"][model_version]
    n_batch_effective = min(n_ctx, n_batch)
    n_ubatch_effective = min(n_batch_effective, n_ubatch)
    n_ctx_effective = pad(n_ctx, p["context_alignment_cells"])
    swa_limit_before_pad = min(
        n_ctx_effective, g["swa_window_cells"] + n_ubatch_effective)
    swa_off_cells = pad(swa_limit_before_pad, p["swa_alignment_cells"])
    swa_cells = n_ctx_effective if swa_full else swa_off_cells
    elements_k = g["kv_heads_per_layer"] * g["key_head_elements"]
    elements_v = g["kv_heads_per_layer"] * g["value_head_elements"]
    types = p["types"]

    def group(layers: int, cells: int) -> dict:
        key = layers * _bytes_for_rows(cells, elements_k, type_k, types)
        value = layers * _bytes_for_rows(cells, elements_v, type_v, types)
        return {
            "layers": layers, "cells": cells,
            "key_logical_bytes_derived": key,
            "value_logical_bytes_derived": value,
            "combined_logical_bytes_derived": key + value,
        }

    glob = group(g["global_layers"], n_ctx_effective)
    swa = group(g["swa_layers"], swa_cells)
    swa_on = group(g["swa_layers"], n_ctx_effective)
    total_k = glob["key_logical_bytes_derived"] + swa["key_logical_bytes_derived"]
    total_v = glob["value_logical_bytes_derived"] + swa["value_logical_bytes_derived"]
    saved = swa_on["combined_logical_bytes_derived"] - swa["combined_logical_bytes_derived"]
    first_swa_saving_at = pad(
        g["swa_window_cells"] + n_ubatch_effective, p["swa_alignment_cells"]
    ) + p["context_alignment_cells"]
    return {
        "status": "llama_kv_source_derived",
        "backend": "llama.cpp",
        "upstream_commit": p["upstream_commit"],
        "binding_version": p["binding_version"],
        "model_version": model_version,
        "model_metadata_status": g["metadata_status"],
        "configuration": {
            "n_ctx_requested": n_ctx,
            "n_ctx_effective_derived": n_ctx_effective,
            "n_batch_requested": n_batch,
            "n_batch_effective_derived": n_batch_effective,
            "n_ubatch_requested": n_ubatch,
            "n_ubatch_effective_derived": n_ubatch_effective,
            "n_seq_max": n_seq_max, "kv_unified": kv_unified,
            "swa_full": swa_full,
            "flash_attention_requested": flash_attention,
            "flash_attention_effective": None,
            "type_k": type_k, "type_v": type_v,
        },
        "geometry": {
            "total_attention_layers": g["total_attention_layers"],
            "global_layers": g["global_layers"],
            "swa_layers": g["swa_layers"],
            "swa_window_cells": g["swa_window_cells"],
            "swa_pattern": g["swa_pattern"],
            "kv_heads_per_layer": g["kv_heads_per_layer"],
            "key_head_elements": g["key_head_elements"],
            "value_head_elements": g["value_head_elements"],
        },
        "padding": {
            "context_cells": p["context_alignment_cells"],
            "swa_cells": p["swa_alignment_cells"],
            "graph_min_cells": p["graph_min_alignment_cells"],
            "graph_kv_n_current": None,
        },
        "global_cache": glob, "swa_cache": swa,
        "key_logical_bytes_derived": total_k,
        "value_logical_bytes_derived": total_v,
        "total_logical_bytes_derived": total_k + total_v,
        "total_logical_mib_derived": round((total_k + total_v) / 1048576, 4),
        "swa_full_off_saves_bytes_derived": saved,
        "swa_full_off_saves_mib_derived": round(saved / 1048576, 4),
        "swa_off_first_saving_at_effective_ctx": first_swa_saving_at,
        "native_kv_buffer_mib_observed": None,
        "native_compute_buffer_mib_observed": None,
        "gpu_fit_verified": False,
        "native_allocation_executed": False,
        "inference_executed": False,
    }
