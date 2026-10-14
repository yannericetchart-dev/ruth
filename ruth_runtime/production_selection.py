"""Production GGUF quality-first selection and strictly gated first unknown probe.

Read-only planning uses Hugging Face pinned metadata, the real host CUDA policy,
and immutable local Git measurements. File bytes > host capacity are a Ruth
SELECTION filter, not proof that CUDA tensor buffers exceed physical VRAM.
Never mark a smaller GGUF as *verified* from a larger quant's success.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

from .candidate_selection import discover_quantizations, load_quantization_policy
from .dependencies import PROJECT_ROOT
from .hardware_eligibility import inspect_host_hardware
from .install_plan import WHEEL_VERSION
from .model_catalog import select_model, CatalogError
from .offline_memory_fit import explain_offline_weights_fit

_MIB = 1024 * 1024
SUPPORTED_PROBES = frozenset(("Q4_K_L", "Q4_K_M"))


def _published_minimal_inference(
    *, root: Path, sha: str, hardware: dict,
) -> dict | None:
    """Reuse only validated exact-runtime minimal-context token evidence."""
    from .profile_contribution import validate_submitted_proposal
    folder = root / "profiles" / "measurements" / sha
    if folder.is_symlink() or not folder.is_dir():
        return None
    for file in sorted(folder.glob("*.json")):
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 200_000:
            raise ValueError("Unsafe published CUDA record")
        validate_submitted_proposal(file, project_root=root)
        payload = json.loads(file.read_text(encoding="utf-8"))
        for v in payload.get("minimal_context_inference", []):
            if (v.get("verified") is True
                and v.get("production_context_qualified") is False
                and v.get("n_ctx") == 128
                and v.get("architecture") == hardware.get("architecture")
                and v.get("driver_version") == hardware.get("driver_version")
                and v.get("cuda_backend") == hardware.get("backend_cuda")
                and v.get("llama_cpp_binding") == WHEEL_VERSION):
                return {"status": "minimal_gpu_tokens_previously_verified",
                        "n_ctx": 128, "source": file.relative_to(root).as_posix(),
                        "production_context_qualified": False}
    return None


def plan_production(
    *, project_root: Path = PROJECT_ROOT,
    authorize_metadata_network: bool = False,
    fetch: Callable[[str], dict] | None = None,
    hardware: dict | None = None,
    discover: Callable[..., dict] = discover_quantizations,
    inspect: Callable[..., dict] = inspect_host_hardware,
    offline_fit: Callable[..., dict] = explain_offline_weights_fit,
) -> dict:
    """No download/weights/probe; only read-only GPU hardware telemetry."""
    root = Path(project_root)
    hw = hardware if hardware is not None else inspect(project_root=root)
    if hw.get("status") != "hardware_eligible_native_backend_unverified":
        return {"status": "production_host_ineligible",
                "hardware": hw, "download_started": False,
                "gpu_allocations_executed": False}
    total = hw.get("total_mib")
    if type(total) is not int or total < 512:
        return {"status": "production_host_capacity_unverified",
                "download_started": False, "gpu_allocations_executed": False}
    policy = load_quantization_policy(root)
    choices = discover(project_root=root, purpose="production",
                       authorize_metadata_network=authorize_metadata_network,
                       fetch=fetch)
    if choices.get("status") != "quantizations_discovered":
        return {"status": "production_source_unverified",
                "source_status": choices.get("status"),
                "source_error": choices.get("error"),
                "download_started": False, "gpu_allocations_executed": False}

    # Remote inventory must match the one immutable allowed repository/revision.
    pinned = select_model("production", project_root=root)
    if choices.get("repo_id") != pinned.repo_id or choices.get("revision") != pinned.revision:
        return {"status": "production_source_mismatch",
                "download_started": False, "gpu_allocations_executed": False}
    rows = []
    first = None
    size_budget = total * _MIB
    matched_runtime = {
        "architecture": hw.get("architecture"),
        "driver_version": hw.get("driver_version"),
        "backend_cuda": hw.get("backend_cuda"),
        "binding_version": WHEEL_VERSION,
        "vram_gib": hw.get("vram_class_gib"),
    }
    for v in choices["quantizations"]:
        quant, size = v["quantization"], v["size_bytes"]
        if v.get("quality_priority_heuristic") is None or v.get("exclusion_reason"):
            state, rationale = "excluded_unranked_or_backend_specific", v.get("exclusion_reason")
            proof = None
        elif type(size) is not int or size <= 0:
            state, rationale, proof = "missing_verified_gguf_size", "Remote LFS size unknown", None
        elif size > size_budget:
            # No CUDA call. This is a conservative Ruth policy exclusion.
            state = "policy_excluded_gguf_file_larger_than_host_vram"
            rationale = "Serialized GGUF exceeds observed host total VRAM; full-GPU probe skipped by policy"
            proof = None
        else:
            proof = None
            confirmed = (_published_minimal_inference(
                root=root, sha=v["sha256"], hardware=hw)
                if v.get("catalog_pin_exists") else None)
            if v.get("catalog_pin_exists"):
                try:
                    proof = offline_fit(project_root=root, quantization=quant,
                                        **matched_runtime)
                except (CatalogError, ValueError):
                    proof = None
            if confirmed is not None:
                state = "minimal_context_gpu_verified_from_git"
                rationale = "Exact GGUF/driver/CUDA minimal-context full GPU token evidence already published"
                proof = confirmed
            elif proof and proof.get("status") == "full_gpu_weights_cannot_fit_nominal_bound":
                state = "measured_full_gpu_weights_incompatible"
                rationale = "Published exact-runtime driver+CUDA+weights measurements exceed VRAM"
            elif proof and proof.get("status") == "weights_possible_context_unverified":
                state = "candidate_requires_runtime_context_and_inference_test"
                rationale = "Measured weights arithmetically possible; KV/compute/inference unverified"
            elif not v.get("download_pin_ready"):
                state = "requires_catalog_pin_before_download"
                rationale = "Remote source verified but local immutable model catalog pin missing"
            elif quant not in SUPPORTED_PROBES:
                state = "requires_backend_probe_support"
                rationale = "Pinned quantization lacks an isolated Ruth native GPU probe"
            else:
                state = "candidate_requires_download_and_gpu_tests"
                rationale = "No decisive exact-runtime full-GPU evidence; probe needed"
        item = {
            "quantization": quant,
            "quality_priority_heuristic": v.get("quality_priority_heuristic"),
            "quality_tier_heuristic": v.get("quality_tier_heuristic"),
            "gguf_size_bytes": size,
            "gguf_size_mib": round(size/_MIB, 2),
            "gguf_sha256": v["sha256"],
            "catalog_pin_exists": v.get("catalog_pin_exists", False),
            "download_pin_ready": v.get("download_pin_ready", False),
            "status": state, "reason": rationale,
            "full_gpu_inference_verified": False,
            "measured_proof": proof,
        }
        rows.append(item)
        if first is None and state in (
            "candidate_requires_download_and_gpu_tests",
            "candidate_requires_runtime_context_and_inference_test",
        ):
            first = quant
    # The discoverer already uses descending heuristic priority. Verify this
    # invariant rather than trusting a provider's arbitrary order.
    rows.sort(key=lambda r: (
        r["quality_priority_heuristic"] is None,
        -(r["quality_priority_heuristic"] or 0),
        r["quantization"],
    ))
    previously_verified = next(
        (r for r in rows if r["status"] == "minimal_context_gpu_verified_from_git"),
        None,
    )
    ordered_testable = [r for r in rows if r["status"] in (
        "candidate_requires_download_and_gpu_tests",
        "candidate_requires_runtime_context_and_inference_test")]
    first = ordered_testable[0]["quantization"] if ordered_testable else None
    # If the best qualified quant was already verified in this exact runtime,
    # avoid retesting smaller/lower-priority variants.
    if previously_verified is not None and (
        first is None or
        previously_verified["quality_priority_heuristic"]
        >= next(r["quality_priority_heuristic"] for r in ordered_testable
                if r["quantization"] == first)
    ):
        first = None
    return {
        "status": "production_selection_ready",
        "model": pinned.version, "purpose": "production",
        "source_repo": pinned.repo_id, "source_revision": pinned.revision,
        "host": {k: hw.get(k) for k in ("gpu_uuid", "gpu_name", "architecture",
                 "compute_capability", "vram_class_gib", "total_mib",
                 "driver_version", "backend_cuda")},
        "gguf_size_policy": "serialized_file_bytes_le_observed_total_vram_bytes",
        "gguf_size_policy_is_not_native_cuda_allocation_proof": True,
        "quality_order_is_heuristic_not_benchmark": True,
        "models": rows,
        "first_candidate_to_test": first,
        "best_previously_verified_minimal_context": (
            previously_verified["quantization"] if previously_verified else None),
        "download_started": False, "gpu_allocations_executed": False,
        "inference_executed": False,
        "smaller_candidates_are_not_automatically_verified": True,
    }


def execute_production_candidate(
    *, plan: dict, project_root: Path = PROJECT_ROOT,
    allow_model_download: bool = False,
    acknowledge_model_license: bool = False,
    allow_gpu_load: bool = False,
    allow_gpu_inference: bool = False,
    download: Callable[..., Any] | None = None,
    load: Callable[..., dict] | None = None,
    infer: Callable[..., Any] | None = None,
    record: Callable[..., dict] | None = None,
    build: Callable[..., dict] | None = None,
    publish_owner: Callable[..., dict] | None = None,
    save_minimal_inference: Callable[..., dict] | None = None,
    hardware: dict | None = None,
) -> dict:
    """Download/test just the first UNKNOWN PINNED candidate, fail closed.

    No retry of historically rejected Q4_K_L. A successful Q4_K_M
    authorizes marking Q4_K_M only; smaller variants remain candidates.
    """
    base = {"status": "production_run_not_authorized", "download_started": False,
            "gpu_allocations_executed": False, "inference_executed": False,
            "git_updated": False}
    if plan.get("status") != "production_selection_ready":
        return {**base, "status": "production_plan_not_ready"}
    candidate = plan.get("first_candidate_to_test")
    if not candidate:
        if plan.get("best_previously_verified_minimal_context"):
            return {**base, "status": "production_previous_minimal_gpu_profile_reused",
                    "candidate": plan["best_previously_verified_minimal_context"],
                    "gpu_inference_verified_historical": True,
                    "current_live_inference_executed": False,
                    "production_qualified": False}
        return {**base, "status": "no_unverified_pinned_candidate_to_probe"}
    row = next((r for r in plan["models"] if r["quantization"] == candidate), None)
    if row is None or not row.get("download_pin_ready") or candidate not in SUPPORTED_PROBES:
        return {**base, "status": "candidate_not_pinned_or_backend_unsupported"}
    if not (allow_model_download and acknowledge_model_license
            and allow_gpu_load and allow_gpu_inference):
        return {**base, "status": "production_explicit_authorization_required",
                "required": ["allow-model-download", "accept-model-license",
                             "allow-gpu-load", "allow-gpu-inference"]}
    root = Path(project_root)
    choice = select_model("production", quantization=candidate, project_root=root)
    if choice.expected_sha256 != row["gguf_sha256"] or choice.repo_id != plan["source_repo"]:
        return {**base, "status": "pin_or_source_changed_before_download"}
    from .model_download import download_pinned_model
    from .load_only_probe import probe_model_load
    from .production_probe import probe_production_candidate
    from .profile_registry import record_local_profile, record_minimal_context_inference
    from .profile_contribution import build_proposal, sync_owner_profile_if_authenticated
    download = download or download_pinned_model
    load = load or probe_model_load
    infer = infer or probe_production_candidate
    record = record or record_local_profile
    build = build or build_proposal
    publish_owner = publish_owner or sync_owner_profile_if_authenticated
    save_minimal_inference = save_minimal_inference or record_minimal_context_inference

    dr = download(choice, project_root=root, allow_download=True,
                  acknowledge_model_license=True)
    if not dr.model_verified:
        return {**base, "status": "production_model_download_or_integrity_failed",
                "candidate": candidate, "download_status": dr.status,
                "download_started": dr.download_started}
    last_report = None
    recording = []
    def publish_measured_profile() -> dict | None:
        if not recording:
            return None
        proposal = build(Path(recording[-1]["profile_file"]))
        return publish_owner(proposal)

    # Keep weights and context distinct, to profile KV allocations independently.
    for mode in ("weights", "context"):
        report = load(project_root=root, quantization=candidate, mode=mode,
                      n_ctx=128, cuda_preinit=True, allow_gpu_load=True)
        if report.get("candidate") != f"gemma-2-9B-it/gguf/{candidate}" or (
            report.get("model_sha256") not in (choice.expected_sha256, None)):
            raise ValueError("Native report model identity mismatch")
        last_report = report
        if report.get("model_verified") is True and report.get("model_sha256") == choice.expected_sha256:
            tmpdir = root / ".ruth-build" / "profiles" / "evidence"
            tmpdir.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(mode="w", suffix=".json", prefix="production-",
                                    encoding="utf-8", dir=tmpdir, delete=True) as temp:
                json.dump({"action": "models", **report}, temp)
                temp.flush()
                recording.append(record(Path(temp.name), project_root=root))
        if report.get("status") != "gpu_load_only_completed" or not report.get("weights_loaded"):
            publication = publish_measured_profile()
            return {**base, "status": "production_native_weights_failed" if mode == "weights"
                    else "production_native_context_failed",
                    "candidate": candidate, "last_native_status": report.get("status"),
                    "download_started": dr.download_started, "gpu_allocations_executed": True,
                    "local_records": recording,
                    "git_publication": publication,
                    "git_updated": bool(publication and publication.get("git_updated")),
                    "next_step": "Inspect failure and Git measurement; never auto-switch quantization blindly"}
        if mode == "context" and not report.get("context_initialized"):
            publication = publish_measured_profile()
            return {**base, "status": "production_native_context_failed",
                    "candidate": candidate, "local_records": recording,
                    "git_publication": publication,
                    "git_updated": bool(publication and publication.get("git_updated")),
                    "gpu_allocations_executed": True}
    result = infer(project_root=root, authorize_gpu_inference=True,
                   quantization=candidate)
    verified = getattr(result, "gpu_inference_verified", False) is True
    publication = None
    minimal_record = None
    if verified and recording:
        # Never store a full-GPU positive status without checking exact GPU
        # identity, layers, native library and token evidence.
        host = hardware if hardware is not None else inspect_host_hardware(project_root=root)
        minimal_record = save_minimal_inference(
            Path(recording[-1]["profile_file"]),
            result.to_dict(), choice=choice, hardware=host)
    if recording:
        publication = publish_measured_profile()
    smaller = [r["quantization"] for r in plan["models"]
               if r.get("quality_priority_heuristic") is not None
               and row.get("quality_priority_heuristic") is not None
               and r["quality_priority_heuristic"] < row["quality_priority_heuristic"]
               and r["status"] not in ("policy_excluded_gguf_file_larger_than_host_vram",
                                      "excluded_unranked_or_backend_specific")]
    return {
        **base,
        "status": ("production_candidate_min_context_gpu_verified" if verified
                   else "production_candidate_inference_unverified"),
        "candidate": candidate, "model_sha256": choice.expected_sha256,
        "download_started": dr.download_started, "gpu_allocations_executed": True,
        "inference_executed": bool(getattr(result, "generated_tokens", 0)),
        "gpu_inference_verified": verified,
        "inference_report": result.to_dict(),
        "local_records": recording,
        "minimal_context_inference_record": minimal_record,
        "git_publication": publication,
        "git_updated": bool(publication and publication.get("git_updated")),
        "smaller_formats_preliminary_size_candidates": smaller if verified else [],
        "smaller_formats_full_gpu_verified": [],
        "production_qualified": False,
        "qualification_note": "Minimal-context GPU tokens verify only this quant and test configuration; "
                              "operational context, FA and resource profile pending",
    }
