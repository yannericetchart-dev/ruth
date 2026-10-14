"""Atomic local model/runtime memory profiles; never push Git from a probe.

All unknown measurements remain null. Local evidence is indexed by pinned GGUF
SHA256 and exact CUDA/backend/driver versions. Community publication is a
separate reviewed PR workflow, not untrusted git push from user machines.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

from .dependencies import PROJECT_ROOT
from .hardware_eligibility import inspect_host_hardware
from .model_catalog import load_catalog
from .model_memory_profiles import _load_evidence, _positive_finite
from .install_plan import WHEEL_VERSION

SCHEMA = "ruth-model-runtime-memory-registry-v1"


def _atomic_write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Refuse symlink profile target")
    raw = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix=".pending-", suffix=".json", delete=False,
        ) as fp:
            name = fp.name
            fp.write(raw)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(name, path)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def _model_from_catalog(root: Path, report: dict) -> tuple[dict, dict]:
    catalog = load_catalog(root)
    for entry in catalog["models"]:
        for variant in entry["variants"]:
            ident = f'{entry["version"]}/{entry["format"]}/{variant["quantization"]}'
            if ident == report.get("candidate"):
                if (variant.get("expected_sha256") != report.get("model_sha256")
                    or variant.get("repo_id") != report.get("source_repository")
                    or variant.get("revision") != report.get("source_revision")
                    or not variant.get("expected_size_bytes")):
                    raise ValueError("Evidence provenance does not match pinned catalog")
                return entry, variant
    raise ValueError("Evidence model/quantization not registered")


def _measurement(value: object) -> float | None:
    return _positive_finite(value)


def _native_buffer(report: dict, kind: str) -> float | None:
    values = [_measurement(row.get("mib"))
              for row in report.get("native_buffers", [])
              if isinstance(row, dict) and row.get("kind") == kind
              and str(row.get("device", "")).startswith("CUDA")]
    valid = [x for x in values if x is not None]
    return round(sum(valid), 2) if valid else None



def initialize_model_profile(
    choice: Any, metadata: dict, *, project_root: Path = PROJECT_ROOT,
) -> dict:
    """Create an UNKNOWN-only local profile after explicit remote pin verification.

    Does NOT accept the model license, download weights or claim CUDA support.
    If a profile exists already, leave every measured value untouched.
    """
    if (metadata.get("status") != "source_metadata_verified_license_pending"
        or metadata.get("sha256") != choice.expected_sha256
        or metadata.get("revision") != choice.revision
        or metadata.get("repo_id") != choice.repo_id
        or metadata.get("size_bytes") != choice.expected_size_bytes):
        raise ValueError("Source metadata is not verified against catalog pins")
    root = Path(project_root)
    target = root / ".ruth-build" / "profiles" / "models" / (
        choice.expected_sha256 + ".json")
    if target.exists():
        if target.is_symlink() or target.stat().st_size > 150_000:
            raise ValueError("Unsafe existing model profile")
        old = json.loads(target.read_text(encoding="utf-8"))
        if old.get("schema") != SCHEMA or (
            old.get("model") or {}).get("gguf_sha256") != choice.expected_sha256:
            raise ValueError("Existing profile does not match pinned GGUF")
        return {"status": "local_profile_already_exists",
                "profile_file": str(target), "git_updated": False}
    profile = {
        "schema": SCHEMA,
        "model": {
            "family": choice.family, "model": choice.model,
            "version": choice.version, "format": choice.format,
            "quantization": choice.quantization,
            "gguf_sha256": choice.expected_sha256,
            "source_repository": choice.repo_id,
            "source_revision": choice.revision,
            "gguf_serialized_bytes": choice.expected_size_bytes,
            "license": choice.license,
        },
        "source_verified": True, "model_license_accepted": False,
        "model_locally_verified": False,
        "native_cuda_weights_mib": None, "runtime_measurements": {},
        "context_profiles": {}, "inference_verified": False,
        "production_qualified": False,
    }
    _atomic_write(target, profile)
    return {"status": "local_profile_initialized_unknown",
            "profile_file": str(target), "git_updated": False,
            "unknowns_preserved_as_null": True,
            "model_license_accepted": False,
            "model_locally_verified": False}



def record_local_profile(
    evidence_file: Path, *, project_root: Path = PROJECT_ROOT,
    hardware: dict | None = None,
) -> dict:
    """Explicit local update only; requires pinned, verified GGUF evidence."""
    root = Path(project_root)
    report = _load_evidence(Path(evidence_file))
    if (report is None or report.get("model_verified") is not True
            or report.get("native_binding_version") != WHEEL_VERSION):
        raise ValueError("Verified model and pinned native binding required")
    entry, variant = _model_from_catalog(root, report)
    hardware = inspect_host_hardware(project_root=root) if hardware is None else hardware
    if (hardware.get("status") != "hardware_eligible_native_backend_unverified"
        or hardware.get("gpu_uuid") != report.get("gpu_uuid")
        or hardware.get("driver_version") != (report.get("gpu_preflight") or {}).get("driver_version")):
        raise ValueError("Host identity, architecture or CUDA eligibility mismatch")
    pre = report.get("gpu_preflight") or {}
    if pre.get("status") != "gpu_ready" or pre.get("processes"):
        raise ValueError("Concurrent GPU use invalidates measurement")

    sha = variant["expected_sha256"]
    target = root / ".ruth-build" / "profiles" / "models" / (sha + ".json")
    identity = {
        "family": entry["family"], "model": entry["model"],
        "version": entry["version"], "format": entry["format"],
        "quantization": variant["quantization"], "gguf_sha256": sha,
        "source_repository": variant["repo_id"], "source_revision": variant["revision"],
        "gguf_serialized_bytes": variant["expected_size_bytes"],
        "license": entry["license"],
    }
    if target.exists():
        if target.stat().st_size > 150_000:
            raise ValueError("Existing profile oversized")
        data = json.loads(target.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA or data.get("model") != identity:
            raise ValueError("Existing local model profile conflicts with GGUF pin")
    else:
        data = {
            "schema": SCHEMA, "model": identity,
            "source_verified": True,
            "native_cuda_weights_mib": None,
            "runtime_measurements": {},
            "context_profiles": {},
            "inference_verified": False,
            "production_qualified": False,
        }

    # Reusing the same immutable probe report must not create a second
    # observation when an earlier Git publication needs to be retried.
    evidence_digest = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seen = data.get("observed_evidence_sha256s", [])
    if not isinstance(seen, list) or any(
        not isinstance(x, str) or len(x) != 64 for x in seen
    ):
        raise ValueError("Malformed existing evidence index")
    if evidence_digest in seen:
        return {
            "status": "local_profile_already_recorded",
            "profile_file": str(target), "model_sha256": sha,
            "git_updated": False, "git_submission_pending": True,
            "observation_duplicated": False,
        }

    gpu_cuda = hardware.get("driver_cuda_max")
    backend_cuda = hardware.get("backend_cuda")
    if not gpu_cuda or not backend_cuda:
        raise ValueError("Exact driver/backend CUDA version unknown")
    # GPU UUID is used for local identity verification, never included in
    # public portable profile keys.
    key = "/".join(("llama.cpp", hardware["architecture"],
                    backend_cuda, gpu_cuda,
                    hardware.get("toolkit_cuda") or "toolkit-unknown",
                    hardware["driver_version"], WHEEL_VERSION))
    record = data["runtime_measurements"].get(key, {
        "architecture": hardware["architecture"],
        "compute_capability": hardware["compute_capability"],
        "tensor_core_generation": hardware["tensor_core_generation"],
        "cuda_backend": backend_cuda, "cuda_driver_max": gpu_cuda,
        "cuda_toolkit": hardware.get("toolkit_cuda"),
        "driver_version": hardware["driver_version"],
        "llama_cpp_binding": WHEEL_VERSION,
        "cuda_init_observed_nvml_delta_mib_max": None,
        "driver_reserved_mib_max": None,
        "cuda_free_before_model_mib": None,
        "weights_buffer_mib_max": None,
        "kv_buffer_mib": None,
        "compute_buffer_mib": None,
        "n_ctx": None,
        "flash_attention": None,
        "kv_type_k": None, "kv_type_v": None,
        "observations": 0,
        "last_load_outcome": None,
    })
    warnings = []
    free_before = _measurement(report.get("cuda_free_mib_after_primary_ctx_init"))
    requested = _measurement(report.get("native_cuda_buffer_mib_reported"))
    if report.get("mode") == "weights" and report.get("weights_loaded") is False:
        measured_weights = requested
    else:
        measured_weights = _native_buffer(report, "model")
    prior_weights = record["weights_buffer_mib_max"]
    if measured_weights is not None:
        if prior_weights is not None and measured_weights > prior_weights:
            warnings.append("native_weights_allocation_exceeds_profiled_max")
        record["weights_buffer_mib_max"] = max(measured_weights,
                                                prior_weights or 0)
        data["native_cuda_weights_mib"] = max(
            record["weights_buffer_mib_max"], data["native_cuda_weights_mib"] or 0)
    mem = pre.get("memory_mib") or {}
    drv_reserved = _measurement(mem.get("reserved"))
    if drv_reserved is not None:
        if (record["driver_reserved_mib_max"] is not None and
            drv_reserved > record["driver_reserved_mib_max"]):
            warnings.append("driver_reservation_increased")
        record["driver_reserved_mib_max"] = max(
            drv_reserved, record["driver_reserved_mib_max"] or 0)
    nvml_free = _measurement(mem.get("free"))
    if free_before is not None:
        record["cuda_free_before_model_mib"] = free_before
    if free_before is not None and nvml_free is not None:
        delta = round(nvml_free-free_before, 2)
        old = record["cuda_init_observed_nvml_delta_mib_max"]
        if old is not None and delta > old:
            warnings.append("cuda_init_observed_delta_increased")
        record["cuda_init_observed_nvml_delta_mib_max"] = max(old or 0, delta)
    record["last_load_outcome"] = report.get("status")
    record["observations"] += 1
    data["runtime_measurements"][key] = record

    if report.get("mode") == "context" and report.get("context_initialized") is True:
        n_ctx = report.get("n_ctx")
        if type(n_ctx) is not int or n_ctx <= 0:
            raise ValueError("Context size missing from native allocation success")
        # Do not assume Flash Attention or KV precision from C API defaults.
        # Both remain null until recorded explicitly by native diagnostics.
        config_key = f"{key}/n_ctx={n_ctx}/flash=unknown/k=unknown/v=unknown"
        data["context_profiles"][config_key] = {
            "n_ctx_requested": n_ctx,
            "kv_size_native": report.get("native_kv_sizes") or None,
            "kv_padding": None,
            "flash_attention": None,
            "kv_type_k": None, "kv_type_v": None,
            "gpu_kv_buffer_mib": _native_buffer(report, "kv"),
            "gpu_compute_buffer_mib": _native_buffer(report, "compute"),
            "backend": "llama.cpp",
            "verified_native_allocations_only": True,
            "inference_verified": False,
        }

    data["observed_evidence_sha256s"] = [*seen, evidence_digest]
    data["model_locally_verified"] = True
    data["model_license_accepted"] = True
    _atomic_write(target, data)
    return {
        "status": "local_profile_updated", "profile_file": str(target),
        "model_sha256": sha, "runtime_key": key,
        "measurement_warnings": warnings,
        "git_updated": False, "git_submission_pending": True,
        "unknowns_preserved_as_null": True,
    }
