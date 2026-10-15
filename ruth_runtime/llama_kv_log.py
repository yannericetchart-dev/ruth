"""Parse pinned llama.cpp native stderr, never conflate log fields with estimates.

This is deliberately revision-scoped; unknown/future log formats stay None.
Per-cache logical K/V sizes are not the physical allocation breakdown.
"""
from __future__ import annotations

import re

_CONTEXT = re.compile(r"\b(?P<key>n_ctx_seq|n_ctx|n_batch|n_ubatch|n_seq_max)\s*=\s*(?P<count>\d+)\b")
_FLASH = re.compile(r"\bflash_attn\s*=\s*(?P<requested>auto|enabled|disabled)\b",re.I)
_KV_CACHE = re.compile(r"\bcreating\s+(?P<group>non-SWA|SWA)\s+KV cache,\s*size\s*=\s*(?P<cells>\d+)\s+cells\b",re.I)
_KV_STATS = re.compile(
    r"\bsize\s*=\s*(?P<total>\d+(?:\.\d+)?)\s+MiB\s*"
    r"\(\s*(?P<cells>\d+)\s+cells,\s*(?P<layers>\d+)\s+layers,\s*"
    r"(?P<seqs>\d+)\s*/\s*(?P<streams>\d+)\s+seqs\).*?"
    r"K\s*\(\s*(?P<key_type>\w+)\s*\)\s*:\s*(?P<key_mib>\d+(?:\.\d+)?)\s*MiB,\s*"
    r"V\s*\(\s*(?P<value_type>\w+)\s*\)\s*:\s*(?P<value_mib>\d+(?:\.\d+)?)\s*MiB",
    re.I,
)
_BUFFER = re.compile(r"(?P<device>CUDA\d+|CPU(?:_[A-Za-z]+)?)\s+(?:KV|compute)\s+buffer size\s*=\s*(?P<mib>\d+(?:\.\d+)?)\s+MiB",re.I)


def parse_llama_context_log(log: str) -> dict:
    if not isinstance(log, str) or len(log) > 2_000_000:
        raise ValueError("Unexpected native log type/size")
    values: dict = {
        "n_ctx_effective_native": None, "n_ctx_seq_effective_native": None,
        "n_batch_effective_native": None, "n_ubatch_effective_native": None,
        "n_seq_max_native": None,
        "flash_attention_native_parameter": None,
        "flash_attention_effective": None,
        "global_kv": None, "swa_kv": None,
        "unassigned_kv_log_records": [],
        "native_log_revision_scoped": True,
    }
    names = {
        "n_ctx": "n_ctx_effective_native",
        "n_ctx_seq": "n_ctx_seq_effective_native",
        "n_batch": "n_batch_effective_native",
        "n_ubatch": "n_ubatch_effective_native",
        "n_seq_max": "n_seq_max_native",
    }
    current_group = None
    for line in log.splitlines():
        kv_new = _KV_CACHE.search(line)
        if kv_new:
            current_group = "global_kv" if kv_new["group"].lower()=="non-swa" else "swa_kv"
            values[current_group] = {"cells": int(kv_new["cells"]),
                                      "native_logical_kv": None,
                                      "backend_kv_buffer_mib": None}
        # Context native settings (do not derive from requested values).
        for m in _CONTEXT.finditer(line):
            values[names[m["key"]]] = int(m["count"])
        flash = _FLASH.search(line)
        if flash:
            values["flash_attention_native_parameter"] = flash["requested"].lower()
        stats = _KV_STATS.search(line)
        if stats:
            kv = {
                "cells": int(stats["cells"]), "layers": int(stats["layers"]),
                "n_seq_max": int(stats["seqs"]), "n_streams": int(stats["streams"]),
                "key_type": stats["key_type"],
                "value_type": stats["value_type"],
                "key_logical_mib_native": float(stats["key_mib"]),
                "value_logical_mib_native": float(stats["value_mib"]),
                "total_logical_mib_native": float(stats["total"]),
            }
            if current_group and values[current_group]:
                values[current_group]["native_logical_kv"] = kv
            else:
                values["unassigned_kv_log_records"].append(kv)
        buf = _BUFFER.search(line)
        if buf and line.lower().find("kv buffer size") >= 0 and current_group:
            if buf["device"].startswith("CUDA"):
                group = values[current_group]
                if group is not None:
                    # A shared CUDA allocation can include multiple tensors.
                    # Keep backend buffer size separate from K and V logical sizes.
                    old = group["backend_kv_buffer_mib"] or 0
                    group["backend_kv_buffer_mib"] = round(old+float(buf["mib"]),2)
    return values
