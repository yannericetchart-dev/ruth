"""Phase 05: discover GGUF quantizations and make a provisional GPU shortlist.

Public Hugging Face metadata ONLY, no weights, no side effects, no CPU fallback.
A size-based shortlist cannot prove model runtime residency, KV usage, quality,
or CUDA backend support. Discovery never silently mutates the pinned catalog.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .model_catalog import (
    CatalogError, GIT_SHA, SHA256, load_catalog, select_model,
    fetch_huggingface_metadata,
)

POLICY_PATH = Path("catalog/quantization_policy.json")
_MIB = 1024 * 1024
_VALID_QUANT = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,24}$")
_ALLOWED_GPU_UUID = re.compile(r"^GPU-[A-Za-z0-9-]{12,64}$")
_QUALITY_TIERS = frozenset({
    "near_original", "very_high", "high", "balanced",
    "compact", "low", "very_low",
})


class CandidateError(ValueError):
    """Unsupported source/model, invalid metadata, or unsafe GPU selection."""


@dataclass(frozen=True)
class GpuMemory:
    uuid: str
    pci_bus: str
    total_mib: int
    free_mib: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_quantization_policy(root: Path = PROJECT_ROOT) -> dict:
    try:
        data = json.loads((Path(root) / POLICY_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        raise CandidateError("Unable to load versioned quantization policy") from exc
    if not isinstance(data, dict) or data.get("schema") != "ruth-quantization-policy-v1":
        raise CandidateError("Unexpected quantization policy schema")
    reserve = data.get("default_provisional_reserve_mib")
    limit = data.get("max_recommended_candidates")
    if type(reserve) is not int or not 0 <= reserve <= 16384 or \
            type(limit) is not int or not 1 <= limit <= 10:
        raise CandidateError("Invalid provisional reserve or candidate limit")
    choices = data.get("quality_policy")
    if not isinstance(choices, list) or not choices:
        raise CandidateError("Empty quantization quality policy")
    names, priorities = set(), set()
    for item in choices:
        if not isinstance(item, dict):
            raise CandidateError("Malformed quality item")
        name = item.get("quantization")
        priority = item.get("priority")
        if not isinstance(name, str) or not _VALID_QUANT.fullmatch(name) \
                or type(priority) is not int or priority <= 0 \
                or name in names or priority in priorities \
                or item.get("tier") not in _QUALITY_TIERS \
                or item.get("cuda_status") != "unverified":
            raise CandidateError("Unsafe, duplicate or unmeasured quality policy")
        names.add(name)
        priorities.add(priority)
    excluded = data.get("excluded_on_cuda")
    if not isinstance(excluded, dict) or any(
        not isinstance(k, str) or not _VALID_QUANT.fullmatch(k) or
        not isinstance(v, str) or not v or k in names
        for k, v in excluded.items()
    ):
        raise CandidateError("Invalid architecture exclusion policy")
    return data


def inspect_single_nvidia_gpu(
    *, run: Callable[..., Any] = subprocess.run,
) -> GpuMemory:
    """Read-only nvidia-smi. Reject multiple GPUs / ambiguous output."""
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,pci.bus_id,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = run(
        command, check=False, capture_output=True, text=True, timeout=10,
    )
    if result.returncode:
        raise CandidateError("Cannot read NVIDIA GPU memory")
    lines = [l.strip() for l in (result.stdout or "").splitlines() if l.strip()]
    if len(lines) != 1:
        raise CandidateError("Exactly one identifiable NVIDIA GPU required")
    parts = [s.strip() for s in lines[0].split(",")]
    if len(parts) != 4 or not _ALLOWED_GPU_UUID.fullmatch(parts[0]) \
            or not parts[1] or not parts[2].isdecimal() \
            or not parts[3].isdecimal():
        raise CandidateError("Invalid NVIDIA memory telemetry")
    total, free = int(parts[2]), int(parts[3])
    if total < 512 or not 0 <= free <= total:
        raise CandidateError("Inconsistent NVIDIA free/total VRAM")
    return GpuMemory(parts[0], parts[1], total, free)


def _get_model_and_variants(root: Path, purpose: str):
    choice = select_model(purpose, project_root=root)
    catalog = load_catalog(root)
    matching = [e for e in catalog["models"]
                if (e["family"], e["model"], e["version"], e["format"]) ==
                (choice.family, choice.model, choice.version, choice.format)]
    if len(matching) != 1:
        raise CandidateError("Model is not uniquely registered")
    item = matching[0]
    variants = item["variants"]
    if any(v["repo_id"] != choice.repo_id for v in variants):
        raise CandidateError("Cross-repository candidate search not supported")
    return choice, item


def _hf_variants(info: dict, choice: Any, item: dict, policy: dict) -> tuple[list, dict]:
    if not isinstance(info, dict) or info.get("private") is True:
        raise CandidateError("Hugging Face model metadata unavailable or private")
    revision = info.get("sha")
    if not isinstance(revision, str) or not GIT_SHA.fullmatch(revision):
        raise CandidateError("Hugging Face revision must be immutable Git SHA")
    if choice.revision is not None and choice.revision != revision:
        raise CandidateError("Hugging Face revision changed from catalog pin")
    license_name = (info.get("cardData") or {}).get("license") \
        if isinstance(info.get("cardData"), dict) else None
    if license_name is not None and license_name != item["license"]:
        raise CandidateError("Hugging Face model license changed")
    if info.get("gated", False) not in (False, None):
        raise CandidateError("Gated source requires separate authorization")
    siblings = info.get("siblings")
    if not isinstance(siblings, list) or not 1 <= len(siblings) <= 3000:
        raise CandidateError("Hugging Face file inventory missing or too large")
    suffix = "-" + choice.quantization + ".gguf"
    if not choice.filename.endswith(suffix):
        raise CandidateError("Cannot derive fixed GGUF family filename prefix")
    prefix = choice.filename[:-len(suffix)]
    re_file = re.compile("^" + re.escape(prefix) + r"-([A-Za-z][A-Za-z0-9_]{1,24})\.gguf$")
    levels = {v["quantization"]: v for v in policy["quality_policy"]}
    existing = {v["quantization"]: v for v in item["variants"]}
    seen: set[str] = set()
    rows = []
    counters = {"gguf_named_variants": 0, "verified_lfs_variants": 0,
                "unranked_or_architecture_specific": 0,
                "missing_lfs_or_size": 0}
    for file in siblings:
        if not isinstance(file, dict) or not isinstance(file.get("rfilename"), str):
            continue
        match = re_file.fullmatch(file["rfilename"])
        if match is None:
            continue
        quant = match.group(1)
        counters["gguf_named_variants"] += 1
        if quant in seen:
            raise CandidateError("Duplicate GGUF variant in Hugging Face response")
        seen.add(quant)
        lfs = file.get("lfs")
        digest = lfs.get("sha256") if isinstance(lfs, dict) else None
        size = file.get("size", lfs.get("size") if isinstance(lfs, dict) else None)
        if type(size) is not int or size < _MIB or \
                not isinstance(digest, str) or not SHA256.fullmatch(digest):
            counters["missing_lfs_or_size"] += 1
            continue  # Never substitute estimated filesize or SHA256.
        pinned = existing.get(quant)
        if pinned is not None and (
            pinned["expected_sha256"] != digest or
            (pinned.get("expected_size_bytes") is not None and
             pinned["expected_size_bytes"] != size) or
            (pinned.get("revision") is not None and
             pinned["revision"] != revision)
        ):
            raise CandidateError("Remote GGUF differs from existing local pin: " + quant)
        tier = levels.get(quant)
        reason = policy["excluded_on_cuda"].get(quant)
        if tier is None:
            counters["unranked_or_architecture_specific"] += 1
        counters["verified_lfs_variants"] += 1
        rows.append({
            "quantization": quant, "filename": file["rfilename"],
            "size_bytes": size, "size_mib": round(size / _MIB, 2),
            "sha256": digest, "revision": revision,
            "quality_priority_heuristic": tier["priority"] if tier else None,
            "quality_tier_heuristic": tier["tier"] if tier else None,
            "cuda_support_verified": False,
            "catalog_pin_exists": pinned is not None,
            "download_pin_ready": bool(
                pinned is not None and
                pinned.get("revision") == revision and
                pinned.get("expected_size_bytes") == size),
            "exclusion_reason": reason or (
                "unranked or architecture-specific quantization"
                if tier is None else None),
            "full_gpu_residency_verified": False,
        })
    if not rows:
        raise CandidateError("No GGUF quantizations with exact LFS SHA256/size")
    rows.sort(key=lambda r: (
        r["quality_priority_heuristic"] is None,
        -(r["quality_priority_heuristic"] or 0), r["filename"],
    ))
    return rows, counters


def discover_quantizations(
    *, project_root: Path = PROJECT_ROOT, purpose: str = "production",
    authorize_metadata_network: bool = False,
    fetch: Callable[[str], dict] | None = None,
) -> dict:
    """No hardware calls, files downloaded, new pins or license acceptance."""
    base = {
        "status": "metadata_network_not_authorized",
        "purpose": purpose, "download_started": False,
        "license_accepted": False, "gpu_inference_executed": False,
        "catalog_modified": False,
    }
    if not authorize_metadata_network:
        return base
    try:
        root = Path(project_root)
        policy = load_quantization_policy(root)
        choice, item = _get_model_and_variants(root, purpose)
        info = (fetch or fetch_huggingface_metadata)(choice.repo_id)
        files, count = _hf_variants(info, choice, item, policy)
        return {
            **base, "status": "quantizations_discovered",
            "model_path": "/".join((choice.family, choice.model, choice.version,
                                   choice.format)),
            "repo_id": choice.repo_id, "revision": info["sha"],
            "license": choice.license,
            "quality_policy": "qualitative_heuristic_not_measured",
            "quantizations": files, "counters": count,
        }
    except (CatalogError, CandidateError, OSError, ValueError,
            TypeError, KeyError) as exc:
        return {
            **base, "status": "quantization_discovery_blocked",
            "error": type(exc).__name__ + ": " + str(exc)[:250],
        }


def recommend_quantization(
    *, project_root: Path = PROJECT_ROOT, purpose: str = "production",
    authorize_metadata_network: bool = False,
    reserve_mib: int | None = None, max_candidates: int | None = None,
    fetch: Callable[[str], dict] | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Quality-ordered size-only prefilter; NEVER a VRAM-fit certification."""
    base = {"status": "metadata_network_not_authorized",
            "download_started": False, "license_accepted": False,
            "gpu_inference_executed": False, "catalog_modified": False,
            "runtime_profile_published": False}
    if not authorize_metadata_network:
        return base
    try:
        root = Path(project_root)
        policy = load_quantization_policy(root)
        reserve = (policy["default_provisional_reserve_mib"]
                   if reserve_mib is None else reserve_mib)
        limit = (policy["max_recommended_candidates"]
                 if max_candidates is None else max_candidates)
        if type(reserve) is not int or not 0 <= reserve <= 16384 or \
                type(limit) is not int or not 1 <= limit <= 10:
            raise CandidateError("Invalid reserve MiB or shortlist limit")
        remote = discover_quantizations(
            project_root=root, purpose=purpose,
            authorize_metadata_network=True, fetch=fetch,
        )
        if remote["status"] != "quantizations_discovered":
            return {**base, "status": remote["status"],
                    "error": remote.get("error")}
        gpu = inspect_single_nvidia_gpu(run=run)
        budget_bytes = max(0, gpu.free_mib - reserve) * _MIB
        considered = []
        for row in remote["quantizations"]:
            reason = row["exclusion_reason"]
            qualifies = reason is None and row["size_bytes"] <= budget_bytes
            considered.append({
                **row, "size_only_prefilter": (
                    "pass_requires_native_test" if qualifies else
                    "architecture_or_unranked" if reason is not None else
                    "exceeds_provisional_size_budget"
                ),
                "provisional_budget_bytes": budget_bytes,
                "estimated_remaining_mib_after_file": round(
                    gpu.free_mib - row["size_bytes"] / _MIB, 2,
                ),
            })
        ordered = [x for x in considered
                   if x["size_only_prefilter"] == "pass_requires_native_test"]
        ordered.sort(key=lambda r: (-r["quality_priority_heuristic"], r["size_bytes"]))
        chosen = ordered[:limit]
        return {
            **base,
            "status": ("provisional_shortlist_ready" if chosen
                       else "no_quantization_passed_provisional_budget"),
            "purpose": purpose, "repo_id": remote["repo_id"],
            "model_path": remote["model_path"], "revision": remote["revision"],
            "quality_policy": remote["quality_policy"],
            "gpu": gpu.to_dict(), "reserve_mib": reserve,
            "reserve_kind": "user_adjustable_unvalidated_budget_not_kv_measurement",
            "size_budget_mib": max(0, gpu.free_mib - reserve),
            "candidates": chosen, "all_quantizations": considered,
            "first_candidate_quantization": (
                chosen[0]["quantization"] if chosen else None
            ),
            "candidate_is_production_qualified": False,
            "next_gate": "pin_license_download_then_real_gpu_offload_kv_and_latency_probes",
            "counters": remote["counters"],
        }
    except (CatalogError, CandidateError, OSError, ValueError,
            TypeError, KeyError, subprocess.SubprocessError) as exc:
        return {**base, "status": "provisional_selection_blocked",
                "error": type(exc).__name__ + ": " + str(exc)[:250]}
