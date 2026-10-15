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
from .install_plan import WHEEL_VERSION, LLAMA_CPP_SUBMODULE_COMMIT
from .context_source_evidence import calculate_context_source, context_source_derivation

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




def record_minimal_context_inference(
    local_profile_file: Path, inference: dict, *,
    choice: Any, hardware: dict,
) -> dict:
    """Persist ONLY verified same-host full CUDA offload with generated tokens.

    A successful minimal-context probe is not production-context qualification.
    No Git side effects; only subsequent sanitized owner Git sync publishes it.
    """
    path = Path(local_profile_file)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 200_000:
        raise ValueError("Missing/unsafe local model profile")
    profile = json.loads(path.read_text(encoding="utf-8"))
    if (profile.get("schema") != SCHEMA or
        (profile.get("model") or {}).get("gguf_sha256") != choice.expected_sha256):
        raise ValueError("Inference GGUF differs from local profile pin")
    if (inference.get("candidate") !=
        f"{choice.version}/gguf/{choice.quantization}" or
        inference.get("gpu_inference_verified") is not True or
        inference.get("full_gpu_offload_verified") is not True or
        inference.get("gpu_process_observed") is not True or
        inference.get("model_verified") is not True or
        inference.get("native_library_verified") is not True or
        inference.get("model_loaded") is not True or
        inference.get("cpu_fallback_selected") is not False or
        inference.get("context_calibration_pending") is not True or
        type(inference.get("generated_tokens")) is not int or
        inference["generated_tokens"] <= 0 or
        type(inference.get("n_ctx")) is not int or
        inference["n_ctx"] <= 0 or
        type(inference.get("offloaded_layers")) is not int or
        inference["offloaded_layers"] <= 0 or
        inference.get("offloaded_layers") != inference.get("total_layers") or
        inference.get("gpu_uuid") != hardware.get("gpu_uuid") or
        (inference.get("gpu_preflight") or {}).get("driver_version")
          != hardware.get("driver_version")):
        raise ValueError("Incomplete native full GPU minimal-context inference proof")
    values = {
        "verified": True,
        "n_ctx": inference["n_ctx"],
        "n_batch": inference.get("n_batch"),
        "n_ubatch": inference.get("n_ubatch"),
        "generated_tokens": inference["generated_tokens"],
        "full_gpu_offload_layers": inference["offloaded_layers"],
        "architecture": hardware.get("architecture"),
        "driver_version": hardware.get("driver_version"),
        "cuda_backend": hardware.get("backend_cuda"),
        "llama_cpp_binding": WHEEL_VERSION,
        "flash_attention": None,
        "kv_type_k": None, "kv_type_v": None,
        "production_context_qualified": False,
    }
    if any(not isinstance(values[k], str) or not values[k] for k in (
        "architecture", "driver_version", "cuda_backend", "llama_cpp_binding")):
        raise ValueError("Unverified native runtime identity")
    entries = profile.setdefault("minimal_context_inference", [])
    if not isinstance(entries, list):
        raise ValueError("Malformed prior inference evidence")
    unique_key = ("n_ctx", "architecture", "driver_version",
                  "cuda_backend", "llama_cpp_binding")
    if not any(all(v.get(k) == values[k] for k in unique_key)
               for v in entries if isinstance(v, dict)):
        entries.append(values)
        _atomic_write(path, profile)
    return {"status": "local_minimal_gpu_inference_verified",
            "model_sha256": choice.expected_sha256,
            "n_ctx": values["n_ctx"], "production_context_qualified": False}

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

    # Historical host snapshot is LOCAL only. It allows a future offline
    # diagnostic to reuse the exact observed GPU total/free values, without
    # querying nvidia-smi or allocating CUDA again. It is never exported
    # by the Git contribution allowlist.
    memory = pre.get("memory_mib") or {}
    snapshot = {
        "architecture": hardware.get("architecture"),
        "compute_capability": hardware.get("compute_capability"),
        "driver_version": hardware.get("driver_version"),
        "vram_class_gib": hardware.get("vram_class_gib"),
        "vram_observed_total_mib": memory.get("total"),
        "nvml_free_before_cuda_mib": memory.get("free"),
        "driver_reserved_mib": memory.get("reserved"),
        "cuda_free_after_primary_ctx_init_mib":
            report.get("cuda_free_mib_after_primary_ctx_init"),
        "historical_measurement": True,
    }
    old_snapshot = data.get("host_snapshot")
    if old_snapshot != snapshot:
        data["host_snapshot"] = snapshot


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
        if data.get("host_snapshot") != snapshot:
            data["host_snapshot"] = snapshot
            _atomic_write(target, data)
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
        options = report.get("context_options_requested")
        native = report.get("native_kv_log")
        source = None
        if options is not None:
            if (not isinstance(options, dict)
                or set(options) != {"n_ctx", "n_batch", "n_ubatch",
                                   "swa_full", "flash_attention", "type_k",
                                   "type_v", "n_seq_max", "kv_unified", "offload_kqv"}
                or options["n_ctx"] != n_ctx
                or options["n_seq_max"] != 1
                or options["kv_unified"] is not False
                or options["offload_kqv"] is not True):
                raise ValueError("Invalid verified C context parameters")
            source = calculate_context_source(
                model_version=entry["version"], project_root=root,
                n_ctx=options["n_ctx"], n_batch=options["n_batch"],
                n_ubatch=options["n_ubatch"], swa_full=options["swa_full"],
                flash_attention=options["flash_attention"],
                type_k=options["type_k"], type_v=options["type_v"])
        if native is not None and not isinstance(native, dict):
            raise ValueError("Malformed native KV diagnostics")
        native = native or {}
        if source is not None:
            actual_ctx = native.get("n_ctx_effective_native")
            if (actual_ctx is not None and
                actual_ctx != source["configuration"]["n_ctx_effective_derived"]):
                raise ValueError("Native context disagrees with pinned backend policy")
            for native_group, expected_group in (
                ("global_kv", "global_cache"), ("swa_kv", "swa_cache")
            ):
                actual = (native.get(native_group) or {}).get("cells")
                if actual is not None and actual != source[expected_group]["cells"]:
                    raise ValueError("Native KV capacity disagrees with pinned backend")
        if source is None:
            config_key = f"{key}/n_ctx={n_ctx}/flash=unknown/k=unknown/v=unknown"
        else:
            opts = source["configuration"]
            config_key = (
                f"{key}/n_ctx={n_ctx}/flash={opts['flash_attention_requested']}"
                f"/k={opts['type_k']}/v={opts['type_v']}"
                f"/swa={'on' if opts['swa_full'] else 'off'}"
                f"/batch={opts['n_batch_requested']}/ubatch={opts['n_ubatch_requested']}"
            )
        glob = native.get("global_kv") or {}
        swa = native.get("swa_kv") or {}
        group_sizes = [v for v in (glob.get("cells"), swa.get("cells"))
                       if type(v) is int]
        def tensor(group, part):
            return (group.get("native_logical_kv") or {}).get(part)
        data["context_profiles"][config_key] = {
            "n_ctx_requested": n_ctx,
            "kv_size_native": group_sizes or report.get("native_kv_sizes") or None,
            "kv_padding": None,
            "flash_attention": (options or {}).get("flash_attention"),
            "kv_type_k": (options or {}).get("type_k"),
            "kv_type_v": (options or {}).get("type_v"),
            "gpu_kv_buffer_mib": _native_buffer(report, "kv"),
            "gpu_compute_buffer_mib": _native_buffer(report, "compute"),
            "backend": "llama.cpp",
            "verified_native_allocations_only": True,
            "inference_verified": False,
            "kv_backend_commit": LLAMA_CPP_SUBMODULE_COMMIT if source else None,
            "native_config": {
                "n_batch_requested": options["n_batch"],
                "n_ubatch_requested": options["n_ubatch"],
                "swa_full": options["swa_full"],
                "flash_attention_requested": options["flash_attention"],
                "flash_attention_effective": native.get("flash_attention_effective"),
                "n_ctx_effective_native": native.get("n_ctx_effective_native"),
                "n_ctx_seq_effective_native": native.get("n_ctx_seq_effective_native"),
                "n_batch_effective_native": native.get("n_batch_effective_native"),
                "n_ubatch_effective_native": native.get("n_ubatch_effective_native"),
                "global_cache_cells_native": glob.get("cells"),
                "swa_cache_cells_native": swa.get("cells"),
                "global_key_logical_mib_native": tensor(glob,"key_logical_mib_native"),
                "global_value_logical_mib_native": tensor(glob,"value_logical_mib_native"),
                "swa_key_logical_mib_native": tensor(swa,"key_logical_mib_native"),
                "swa_value_logical_mib_native": tensor(swa,"value_logical_mib_native"),
            } if source else None,
            "kv_source_derivation": context_source_derivation(source) if source else None,
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
