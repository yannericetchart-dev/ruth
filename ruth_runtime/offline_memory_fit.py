"""Offline, fail-closed full-GPU weights fit from committed Git profile evidence.

No nvidia-smi, CUDA libraries, native inference, network or live memory reads.
A negative bound is valid if even nominal physical VRAM cannot accommodate
the measured driver reservation, CUDA initialization delta and model buffer.
A positive arithmetic margin is NEVER a production-compatibility verdict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dependencies import PROJECT_ROOT
from .hardware_eligibility import load_architecture_policy
from .install_plan import WHEEL_VERSION
from .model_catalog import select_model
from .profile_contribution import validate_submitted_proposal
from .model_memory_profiles import _positive_finite


def _runtime_local_identity(root: Path, sha: str) -> list[dict]:
    """Use only already-saved local model profile, not the current GPU."""
    path = root / ".ruth-build" / "profiles" / "models" / (sha + ".json")
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 200_000:
        return []
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("schema") != "ruth-model-runtime-memory-registry-v1"
        or (data.get("model") or {}).get("gguf_sha256") != sha):
        return []
    rows = data.get("runtime_measurements")
    return [r for r in rows.values() if isinstance(r, dict)] if isinstance(rows, dict) else []


def explain_offline_weights_fit(
    *, project_root: Path = PROJECT_ROOT,
    quantization: str = "Q4_K_L",
    architecture: str | None = None,
    vram_gib: int | None = None,
    driver_version: str | None = None,
    backend_cuda: str | None = None,
    binding_version: str | None = None,
) -> dict:
    """Explain a specific pinned GGUF and nominal VRAM class from Git data."""
    root = Path(project_root)
    choice = select_model(purpose="production", quantization=quantization)
    sha = choice.expected_sha256
    policy = load_architecture_policy(root)
    locally_known = _runtime_local_identity(root, sha)
    source_identity = {
        "architecture": architecture,
        "driver_version": driver_version,
        "cuda_backend": backend_cuda,
        "llama_cpp_binding": binding_version,
    }
    # Ambiguous local history must not turn into an assumed host identity.
    for field, value in source_identity.items():
        if value is not None:
            continue
        candidates = {r.get(field) for r in locally_known
                      if isinstance(r.get(field), str) and r.get(field)}
        if len(candidates) == 1:
            source_identity[field] = next(iter(candidates))
    arch_id = source_identity["architecture"]
    drv = source_identity["driver_version"]
    backend = source_identity["cuda_backend"]
    binding = source_identity["llama_cpp_binding"]
    classes = {a["id"]: a for a in policy["architectures"]}
    if arch_id not in classes:
        return _unverified("architecture_unknown_or_unsupported", sha, source_identity)
    arch = classes[arch_id]
    if vram_gib is None:
        # Newer local registries can retain a nominal VRAM class without
        # needing a fresh nvidia-smi call.
        snapshots = set()
        snapshot = root / ".ruth-build" / "profiles" / "models" / (sha + ".json")
        if snapshot.is_file() and not snapshot.is_symlink():
            import json
            local = json.loads(snapshot.read_text(encoding="utf-8"))
            cls = (local.get("host_snapshot") or {}).get("vram_class_gib")
            if type(cls) is int:
                snapshots.add(cls)
        if len(snapshots) == 1:
            vram_gib = next(iter(snapshots))
    if type(vram_gib) is not int or vram_gib not in arch["vram_classes_gib"]:
        return _unverified("host_vram_class_missing_or_forbidden", sha, source_identity)
    if not all(isinstance(x, str) and x for x in (drv, backend, binding)):
        return _unverified("exact_runtime_identity_missing", sha, source_identity)

    folder = root / "profiles" / "measurements" / sha
    if not folder.is_dir() or folder.is_symlink():
        return _unverified("git_profile_not_present_locally", sha, source_identity)
    reports = []
    for file in sorted(folder.glob("*.json")):
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 200_000:
            raise ValueError("Unsafe committed profile")
        validate_submitted_proposal(file, project_root=root)
        import json
        payload = json.loads(file.read_text(encoding="utf-8"))
        if payload["model"]["gguf_sha256"] != sha:
            raise ValueError("Wrong model SHA in committed memory record")
        for row in payload["runtime_measurements"]:
            if (row["architecture"] == arch_id
                and row["driver_version"] == drv
                and row["cuda_backend"] == backend
                and row["llama_cpp_binding"] == binding):
                reports.append((file.relative_to(root).as_posix(), row))
    if not reports:
        return _unverified("no_exact_architecture_cuda_driver_binding_evidence",
                           sha, source_identity)
    reserved = [_positive_finite(r.get("driver_reserved_mib_max")) for _,r in reports]
    init = [_positive_finite(r.get("cuda_init_observed_nvml_delta_mib_max")) for _,r in reports]
    weights = [_positive_finite(r.get("weights_buffer_mib_max")) for _,r in reports]
    if any(not vals or any(v is None for v in vals) for vals in (reserved, init, weights)):
        return _unverified("incomplete_native_memory_measurements", sha, source_identity)
    # Per key maxima are a conservative envelope of observations, not proof of
    # simultaneous memory usage. It supports a negative capacity bound only.
    nominal_total = vram_gib * 1024
    driver_mib = max(reserved)
    init_mib = max(init)
    weights_mib = max(weights)
    available = round(nominal_total - driver_mib - init_mib, 2)
    deficit = round(weights_mib - available, 2)
    if deficit > 0:
        status, compatible = "full_gpu_weights_cannot_fit_nominal_bound", False
        conclusion = (
            "Insufficient nominal physical VRAM for the pinned model weights "
            "under the matched measured driver/CUDA runtime overhead. "
            "KV cache and compute buffers were never required for this rejection."
        )
    else:
        status, compatible = "weights_possible_context_unverified", None
        conclusion = (
            "Nominal arithmetic does not exclude the weights, but actual CUDA "
            "allocatability, KV cache, Flash Attention and inference remain unverified."
        )
    return {
        "status": status, "model": choice.version,
        "quantization": quantization, "gguf_sha256": sha,
        "hardware": {
            "architecture": arch_id, "tensor_core_generation": arch["tensor_core_generation"],
            "vram_class_gib": vram_gib, "vram_nominal_total_mib": nominal_total,
            "driver_version": drv, "cuda_backend": backend,
            "llama_cpp_binding": binding,
            "hardware_identity_source": "saved_local_profile_and_explicit_vram_class",
            "live_host_status": "not_checked",
        },
        "memory": {
            "driver_reserved_mib": driver_mib,
            "cuda_init_observed_nvml_to_cuda_delta_mib": init_mib,
            "weights_cuda_buffer_mib": weights_mib,
            "vram_nominal_total_mib": nominal_total,
            "vram_estimated_available_for_weights_mib": available,
            "vram_weights_deficit_mib": max(0., deficit),
            "measured_live_cuda_free_mib": None,
            "kv_buffer_mib": None,
            "compute_buffer_mib": None,
        },
        "compatible_for_full_gpu_weights": compatible,
        "production_qualified": False, "conclusion": conclusion,
        "evidence_files": sorted({path for path,_ in reports}),
        "driver_and_cuda_values_are_historical": True,
        "cuda_init_value_is_cross_api_delta_not_exact_context_allocation": True,
        "nominal_class_is_upper_bound_not_observed_actual_total": True,
        "no_gpu_queries": True, "no_gpu_allocations": True,
        "no_inference": True, "network_used": False,
    }


def _unverified(reason: str, sha: str, identity: dict) -> dict:
    return {
        "status": "offline_fit_unverified", "reason": reason,
        "gguf_sha256": sha, "cached_identity": identity,
        "compatible_for_full_gpu_weights": None,
        "production_qualified": False,
        "no_gpu_queries": True, "no_gpu_allocations": True,
        "no_inference": True, "network_used": False,
    }
