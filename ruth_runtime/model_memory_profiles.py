"""Read-only CUDA memory profile matrix: measured evidence versus class projections.

A nominal VRAM class is not a runtime guarantee. Never promote a size-based
projection or a load-only success to an inference-qualified production profile.
No CUDA initialization, files downloaded, processes killed or GPU allocations.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable

from .candidate_selection import GpuMemory, inspect_single_nvidia_gpu
from .dependencies import PROJECT_ROOT
from .gpu_preflight import inspect_gpu_preflight
from .install_plan import WHEEL_VERSION
from .model_catalog import load_catalog

CAPACITY_FILE = Path("catalog/gpu_capacity_profiles.json")
_MIB = 1024 * 1024
_CARD = re.compile(r"card[0-9]+$")
_PCI_BUS = re.compile(r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])$")


def _canonical_pci_bus(raw: str) -> str | None:
    match = _PCI_BUS.fullmatch(raw)
    if match is None:
        return None
    domain, bus, device, function = match.groups()
    return f"{int(domain,16):04x}:{bus.lower()}:{device.lower()}.{function}"


def inspect_gpu_drm(pci_bus: str, *, sysroot: Path = Path("/sys")) -> dict:
    """Detect whether an NVIDIA DRM card is associated with this PCI GPU.

    A loaded nvidia_drm module is not itself proof that it uses this GPU.
    This is a sysfs-only observation; no module removal or driver changes.
    """
    expected_bus = _canonical_pci_bus(pci_bus)
    if expected_bus is None:
        return {"status": "drm_telemetry_unverified", "module_loaded": None,
                "associated_cards": []}
    root = Path(sysroot)
    base = root / "class/drm"
    mod = root / "module/nvidia_drm"
    if not base.is_dir() or not (root / "module").is_dir():
        return {"status": "drm_telemetry_unverified", "module_loaded": None,
                "associated_cards": []}
    loaded = mod.is_dir()
    cards = []
    try:
        for path in base.iterdir():
            if _CARD.fullmatch(path.name) and _canonical_pci_bus((path / "device").resolve().name) == expected_bus:
                cards.append(path.name)
    except OSError:
        return {"status": "drm_telemetry_unverified", "module_loaded": loaded,
                "associated_cards": []}
    return {
        "status": ("drm_active_on_selected_gpu" if cards and loaded else
                   "drm_module_loaded_no_selected_gpu_card" if loaded else
                   "drm_not_loaded"),
        "module_loaded": loaded,
        "associated_cards": sorted(cards),
    }


def _positive_finite(x: object) -> float | None:
    if type(x) not in (float, int) or isinstance(x, bool):
        return None
    v = float(x)
    return round(v, 2) if 0 < v < 1_000_000 else None


def _load_evidence(path: Path | None) -> dict | None:
    if path is None:
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 150_000:
        raise ValueError("Unsafe or oversized explicit evidence JSON")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("action") != "models" \
            or data.get("mode") not in ("weights", "context") \
            or data.get("inference_executed") is not False \
            or data.get("generated_tokens") != 0:
        raise ValueError("Not a valid no-inference Ruth CUDA load report")
    return data


def _match_evidence(evidence: dict | None, model: dict) -> tuple[dict | None, str]:
    if not evidence:
        return None, "no_local_evidence"
    if evidence.get("candidate") != model["id"]:
        return None, "different_gguf_candidate"
    if evidence.get("model_sha256") != model["sha256"]:
        return None, "missing_or_mismatched_gguf_sha256"
    if evidence.get("native_binding_version") != WHEEL_VERSION:
        return None, "different_or_unknown_native_runtime_version"
    native = None
    buffers = evidence.get("native_buffers") or []
    # Never interpret a later KV/compute OOM as a model-weights requirement.
    if evidence.get("mode") == "weights" and evidence.get("weights_loaded") is False:
        native = _positive_finite(evidence.get("native_cuda_buffer_mib_reported"))
    if native is None:
        if not isinstance(buffers, list):
            return None, "malformed_native_buffers"
        cuda_weights = [_positive_finite(x.get("mib"))
                        for x in buffers if isinstance(x, dict)
                        and x.get("kind") == "model"
                        and str(x.get("device", "")).startswith("CUDA")]
        native = max((v for v in cuda_weights if v is not None), default=None)
    if native is None:
        return None, "no_native_weights_buffer_measurement"
    pre = evidence.get("gpu_preflight") or {}
    if not isinstance(pre, dict):
        return None, "malformed_gpu_evidence"
    mem = pre.get("memory_mib") or {}
    if not isinstance(mem, dict):
        return None, "malformed_gpu_memory_evidence"
    before = _positive_finite(mem.get("free"))
    after = _positive_finite(evidence.get("cuda_free_mib_after_primary_ctx_init"))
    # Snapshot delta is cross-API/time: NOT a universal CUDA context size.
    delta = round(before - after, 2) if before is not None and after is not None else None
    buff = buffers if isinstance(buffers, list) else []
    def allocated(kind: str) -> float | None:
        values = [_positive_finite(item.get("mib"))
                  for item in buff if isinstance(item, dict)
                  and item.get("kind") == kind
                  and str(item.get("device", "")).startswith("CUDA")]
        valid = [v for v in values if v is not None]
        return round(sum(valid), 2) if valid else None
    return {"native_weights_buffer_mib": native,
            "driver_reserved_mib": mem.get("reserved"),
            "nvml_free_before_cuda_mib": before,
            "observed_nvml_to_cuda_delta_mib": delta,
            "native_kv_gpu_buffer_mib": allocated("kv"),
            "native_compute_gpu_buffer_mib": allocated("compute"),
            "evidence_status": evidence.get("status"),
            "gpu_uuid": pre.get("gpu_uuid"),
            "driver_version": pre.get("driver_version"),
            "cuda_free_after_init_mib": _positive_finite(
                evidence.get("cuda_free_mib_after_primary_ctx_init")),
            "weights_loaded": evidence.get("weights_loaded") is True,
            "context_initialized": evidence.get("context_initialized") is True,
            "full_gpu_offload_observed": evidence.get("full_gpu_offload_observed") is True,
            "n_ctx": evidence.get("n_ctx"),
            "native_kv_sizes": evidence.get("native_kv_sizes") or [],
            "native_buffers": evidence.get("native_buffers") or [],
            }, "validated_pinned_evidence"


def _models(root: Path) -> list[dict]:
    data = load_catalog(root)
    result = []
    for entry in data["models"]:
        for variant in entry["variants"]:
            ident = f'{entry["version"]}/{entry["format"]}/{variant["quantization"]}'
            result.append({
                "id": ident, "family": entry["family"], "model": entry["model"],
                "version": entry["version"], "quantization": variant["quantization"],
                "sha256": variant["expected_sha256"],
                "gguf_size_bytes": variant.get("expected_size_bytes"),
                "license": entry["license"],
            })
    return result


def build_profile_matrix(
    *, project_root: Path = PROJECT_ROOT,
    evidence_file: Path | None = None,
    check_host: bool = False,
    force: bool = False,
    gpu_inspector: Callable[..., GpuMemory] = inspect_single_nvidia_gpu,
    preflight: Callable[..., Any] = inspect_gpu_preflight,
    drm_inspector: Callable[..., dict] = inspect_gpu_drm,
) -> dict:
    """Read-only profile listing and fail-closed hardware-fit planning.

    'force' selects an EXPERIMENT recommendation only. Execution still needs
    a separate explicit probe command and its GPU/model gates.
    """
    root = Path(project_root)
    policy = json.loads((root / CAPACITY_FILE).read_text(encoding="utf-8"))
    classes = policy.get("vram_classes_gib")
    if policy.get("schema") != "ruth-gpu-capacity-classes-v1" or \
            not isinstance(classes, list) or not classes or \
            any(type(x) is not int or not 1 <= x <= 512 for x in classes) or \
            len(set(classes)) != len(classes):
        raise ValueError("Invalid GPU capacity class policy")
    evidence = _load_evidence(evidence_file)
    models = _models(root)
    host = None
    if check_host:
        gpu = gpu_inspector()
        gate = preflight(gpu.uuid)
        drm = drm_inspector(gpu.pci_bus)
        host = {
            "gpu_uuid": gpu.uuid, "pci_bus": gpu.pci_bus,
            "total_mib": gpu.total_mib, "free_mib": gpu.free_mib,
            "gpu_preflight": gate.to_dict(), "drm": drm,
            "exclusivity_verified": gate.status == "gpu_ready",
            "compute_only_verified": (gate.status == "gpu_ready"
                                      and drm["status"] == "drm_not_loaded"),
        }
    records = []
    for model in models:
        measured, state = _match_evidence(evidence, model)
        weight_mib = measured["native_weights_buffer_mib"] if measured else None
        tiers = []
        for gib in classes:
            total = gib * 1024
            if weight_mib is not None and weight_mib > total:
                fit = "insufficient_nominal_vram_for_measured_weights"
            elif weight_mib is not None:
                fit = "weights_physically_possible_unverified"
            else:
                fit = "native_memory_profile_pending"
            tiers.append({
                "nominal_gib": gib, "physical_mib": total,
                "status": fit, "full_runtime_compatible": False,
            })
        host_status = None
        if host:
            if not host["exclusivity_verified"]:
                host_status = "host_gpu_busy_or_unverified"
            elif host["drm"]["status"] == "drm_active_on_selected_gpu":
                host_status = "gpu_drm_active_review_compute_only_policy"
            elif weight_mib is None:
                host_status = "host_native_memory_unmeasured"
            elif (measured["gpu_uuid"] != host["gpu_uuid"] or
                  measured["driver_version"] != host["gpu_preflight"].get("driver_version")):
                host_status = "host_evidence_identity_mismatch"
            elif (measured["cuda_free_after_init_mib"] is not None and
                  measured["cuda_free_after_init_mib"] < weight_mib):
                host_status = "host_measured_weights_cannot_fit"
            elif measured["weights_loaded"] and measured["context_initialized"] \
                    and measured["full_gpu_offload_observed"]:
                host_status = "host_context_allocation_observed_inference_unverified"
            elif measured["weights_loaded"] and measured["full_gpu_offload_observed"]:
                host_status = "host_weights_allocated_context_unverified"
            else:
                host_status = "host_requires_native_allocation_verification"
        records.append({
            "id": model["id"], "family": model["family"],
            "quantization": model["quantization"],
            "gguf_size_bytes": model["gguf_size_bytes"],
            "native_evidence_status": state,
            "measured_weights_buffer_mib": weight_mib,
            "source_evidence": measured,
            "capacity_classes": tiers,
            "host_status": host_status,
            "experiment_force_requested": force,
            "experiment_allowed_in_separate_probe": bool(
                force and host and host["exclusivity_verified"] and
                host["drm"]["status"] == "drm_not_loaded" and
                model["family"] == "gemma" and
                model["quantization"] in ("Q4_K_L", "Q4_K_M")),
            "production_qualified": False,
        })
    return {
        "schema": "ruth-model-memory-profiles-v1",
        "status": "host_profile_matrix_ready" if host else "profile_catalog_ready",
        "capacity_classes_gib": classes, "profiles": records,
        "host": host,
        "force_is_selection_only": True, "gpu_inference_executed": False,
        "gpu_allocations_executed": False,
        "no_cpu_fallback": True, "download_started": False,
        "qualification_policy": "measured_lower_bounds_and_host_fit_not_production_certification",
    }
