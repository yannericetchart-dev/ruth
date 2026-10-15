"""One Git canonical profile per model, with versioned variant/runtime observations.

Pure, offline, deterministic. The sole public source of truth is
profiles/models/<family>/<model>/<version>/profile.json, not historical
content-digest publication files. Raw host evidence remains local.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .dependencies import PROJECT_ROOT
from .model_catalog import load_catalog

SCHEMA = "ruth-model-profile-v2"
SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
WEIGHT_FORMATS = ("F32", "F16", "BF16")
_SOURCE = {"weight_types": ["F32", "F16", "BF16"],
           "gguf_min_n_ctx_requested": 256, "context_alignment": 256,
           "context_step": 256,
           "scope": "single_sequence_pinned_llama_cpp",
           "n_ctx_256_profile_batch": None,
           "n_ctx_256_profile_ubatch": None,
           "fixed_kv_type_k": "F16", "fixed_kv_type_v": "F16",
           "fixed_flash_attention": "off", "fixed_swa_full": True}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _model_entry(root: Path, family: str, model: str, version: str) -> dict:
    matches = [entry for entry in load_catalog(root)["models"]
               if (entry["family"], entry["model"], entry["version"]) ==
               (family, model, version)]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous model identity")
    return matches[0]


def profile_path(family: str, model: str, version: str,
                 project_root: Path = PROJECT_ROOT) -> Path:
    if not all(isinstance(s, str) and SEGMENT.fullmatch(s)
               for s in (family, model, version)):
        raise ValueError("Unsafe canonical model profile path")
    return Path(project_root) / "profiles" / "models" / family / model / version / "profile.json"


def create_profile(entry: dict) -> dict:
    variants = {}
    for var in entry["variants"]:
        q = var["quantization"].upper()
        variants[q] = {
            "format": q,
            "precision_family": "floating" if q in WEIGHT_FORMATS else "quantized",
            "gguf_sha256": var["expected_sha256"],
            "file_bytes": var.get("expected_size_bytes"),
            "source_repository": var["repo_id"],
            "source_revision": var["revision"],
            "filename": var["filename"],
            "runtime_observations": {},
            "context_observations": {},
            "minimal_inference_observations": {},
        }
    # Model parameters are source defaults, NOT native proof for every GGUF.
    return {
        "schema": SCHEMA,
        "model": {
            "family": entry["family"], "model": entry["model"],
            "version": entry["version"], "license": entry["license"],
            "upstream_repo": entry["upstream_repo"],
        },
        "architecture": {
            "status": "unknown", "global_layers": None,
            "swa_layers": None, "swa_window": None,
            "kv_heads": None, "key_head_dim": None, "value_head_dim": None,
        },
        "backend_policies": {"llama.cpp": {
            "status": "source_derived",
            "pinned_revision": "0c1e57098bba43ac29e6e3b677cdceebdd22334f",
            **copy.deepcopy(_SOURCE),
        }},
        "variants": variants,
        "selection": {
            "quality_policy": "quality_heuristic_within_model_only",
            "cross_model_quality_score": None,
            "precision_preference": ["F32", "BF16", "F16"],
        },
        "provenance": {"historical_migration_completed": False},
    }


def _verify_profile(profile: dict, entry: dict) -> None:
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise ValueError("Unexpected canonical model profile schema")
    model = profile.get("model") or {}
    if (model.get("family"), model.get("model"), model.get("version")) != (
            entry["family"], entry["model"], entry["version"]):
        raise ValueError("Canonical profile identity mismatch")
    variants = profile.get("variants")
    if not isinstance(variants, dict):
        raise ValueError("Missing canonical model variants")
    for item in entry["variants"]:
        existing = variants.get(item["quantization"].upper())
        if (not isinstance(existing, dict)
            or existing.get("gguf_sha256") != item["expected_sha256"]
            or existing.get("file_bytes") != item.get("expected_size_bytes")
            or existing.get("source_repository") != item["repo_id"]
            or existing.get("source_revision") != item["revision"]):
            raise ValueError("Canonical variant does not match trusted catalog pin")


def load_profile(family: str, model: str, version: str,
                 *, project_root: Path = PROJECT_ROOT) -> dict:
    """Read Git canonical profile, or create an UNKNOWN skeleton in memory.

    No CUDA, internet or hardware access; caller decides if/when to commit.
    """
    root = Path(project_root)
    entry = _model_entry(root, family, model, version)
    path = profile_path(family, model, version, root)
    if not path.exists():
        return create_profile(entry)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 3_000_000:
        raise ValueError("Unsafe canonical profile")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _verify_profile(payload, entry)
    return payload


def _merge_index(index: dict, signature: dict, row: dict) -> bool:
    # An observation is keyed by a stable runtime/config signature, but
    # different measurements for that configuration retain separate evidence.
    key = hashlib.sha256(_canonical(signature).encode("utf-8")).hexdigest()[:24]
    matches = [v for v in index.values() if v.get("identity") == signature]
    if len(matches) > 1:
        raise ValueError("Duplicated canonical observation identity")
    bucket = matches[0] if matches else index.setdefault(
        key, {"identity": signature, "observations": []})
    if bucket.get("identity") != signature or not isinstance(bucket.get("observations"), list):
        raise ValueError("Canonical observation identity collision")
    digest = hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest()
    if any(_canonical(v.get("data")) == _canonical(row) for v in bucket["observations"]):
        return False
    if len(bucket["observations"]) >= 64:
        raise ValueError("Canonical configuration observation limit exceeded")
    bucket["observations"].append({"sha256": digest, "data": row})
    bucket["observations"].sort(key=lambda a: a.get("sha256") or "")
    return True


def merge_published_measurement(profile: dict, measurement: dict) -> tuple[dict, bool]:
    """Add sanitized validated v1 Git observation to canonical profile.

    Caller MUST call validate_submitted_proposal() for new external input;
    this routine additionally checks immutable source and exact GGUF pins.
    """
    model = measurement.get("model") or {}
    if (model.get("family"), model.get("model"), model.get("version")) != (
        profile["model"]["family"], profile["model"]["model"],
        profile["model"]["version"]):
        raise ValueError("Measurement refers to another model")
    if measurement.get("schema") != "ruth-community-model-memory-proposal-v1":
        raise ValueError("Measurement not validated v1 schema")
    quant = str(model.get("quantization") or "").upper()
    variant = (profile.get("variants") or {}).get(quant)
    if (variant is None
        or variant["gguf_sha256"] != model.get("gguf_sha256")
        or variant["file_bytes"] != model.get("gguf_serialized_bytes")
        or variant["source_repository"] != model.get("source_repository")
        or variant["source_revision"] != model.get("source_revision")):
        raise ValueError("Unpinned or inconsistent GGUF measurement")
    updated = copy.deepcopy(profile)
    node = updated["variants"][quant]
    changed = False
    rows = measurement.get("runtime_measurements") or []
    for row in rows:
        signature = {k: row.get(k) for k in
                     ("architecture", "compute_capability", "cuda_backend",
                      "cuda_driver_max", "driver_version", "llama_cpp_binding")}
        changed |= _merge_index(node["runtime_observations"], signature, row)
    for ctx in measurement.get("context_profiles") or []:
        native = ctx.get("native_config") or {}
        signature = {
            "architecture": ctx.get("architecture"),
            "cuda_backend": ctx.get("cuda_backend"),
            "driver_version": ctx.get("driver_version"),
            "llama_cpp_binding": ctx.get("llama_cpp_binding"),
            "backend_revision": ctx.get("kv_backend_commit"),
            "n_ctx_requested": ctx.get("n_ctx_requested"),
            "flash_attention": ctx.get("flash_attention"),
            "type_k": ctx.get("kv_type_k"),
            "type_v": ctx.get("kv_type_v"),
            "swa_full": native.get("swa_full"),
            "n_batch": native.get("n_batch_requested"),
            "n_ubatch": native.get("n_ubatch_requested"),
        }
        changed |= _merge_index(node["context_observations"], signature, ctx)
    for inf in measurement.get("minimal_context_inference") or []:
        signature = {k: inf.get(k) for k in
                     ("architecture", "cuda_backend", "driver_version",
                      "llama_cpp_binding", "n_ctx", "n_batch", "n_ubatch",
                      "flash_attention", "kv_type_k", "kv_type_v")}
        changed |= _merge_index(node["minimal_inference_observations"], signature, inf)
    return updated, changed


def resolve_model_profile(purpose: str, family: str, model: str, version: str,
                          *, project_root: Path = PROJECT_ROOT) -> dict:
    """Common production/testing lookup: profiles do NOT depend on purpose."""
    if purpose not in ("production", "test"):
        raise ValueError("Unsupported model purpose")
    profile = load_profile(family, model, version, project_root=project_root)
    return {"status": "canonical_profile_loaded" if profile["provenance"].get(
        "historical_migration_completed") else "canonical_profile_created_in_memory",
        "path": str(profile_path(family, model, version, project_root)),
        "profile": profile, "purpose": purpose,
        "gpu_allocations_executed": False}


def load_probe_config(profile: dict, *, n_ctx: int, n_batch: int, n_ubatch: int,
                      type_k: str="F16", type_v: str="F16",
                      flash_attention: str="off", swa_full: bool=True) -> dict:
    """Only loading baseline is currently qualified as a profiling domain."""
    backend = profile["backend_policies"]["llama.cpp"]
    if (type(n_ctx) is not int or n_ctx < backend["gguf_min_n_ctx_requested"]
        or n_ctx % backend["context_step"] != 0):
        raise ValueError("llama.cpp profile context must be a multiple of 256, starting at 256")
    if (type(n_batch) is not int or type(n_ubatch) is not int
        or not 1 <= n_ubatch <= n_batch <= n_ctx):
        raise ValueError("Invalid batch/microbatch for profiling")
    if ((type_k, type_v, flash_attention, swa_full)
        != ("F16", "F16", "off", True)):
        raise ValueError("Load phase is fixed to KV F16/F16, Flash OFF, SWA ON")
    return {"n_ctx": n_ctx, "n_batch": n_batch, "n_ubatch": n_ubatch,
            "type_k": type_k, "type_v": type_v, "flash_attention": flash_attention,
            "swa_full": swa_full,
            "n_ctx_effective_derived": n_ctx,
            "profile_geometry_status": "source_derived"}
