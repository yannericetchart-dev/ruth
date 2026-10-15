"""Persist one native load probe locally and opportunistically sync owner Git.

This function never starts CUDA, downloads a GGUF, or performs inference.
The raw evidence remains LOCAL and is never uploaded. Git only receives the
sanitized strict allowlist produced by build_proposal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .dependencies import PROJECT_ROOT

_SHA = re.compile(r"^[0-9a-f]{64}$")
_MEASURED_STATUSES = frozenset(("gpu_load_only_completed", "gpu_load_only_failed"))


def record_probe_evidence(
    report: dict,
    *,
    project_root: Path = PROJECT_ROOT,
    sync_git: bool = True,
) -> dict:
    """Retain a successful/diagnosed native probe, even if Git publication fails.

    Never mark a failed allocation as an inference or context pass. A failed
    sync is report metadata, not a reason to discard measured evidence.
    """
    result = {
        "status": "native_probe_not_recorded",
        "local_profile_update": None,
        "git_publication": None,
        "runtime_profile_published": False,
    }
    if (
        not isinstance(report, dict)
        or report.get("action") not in (None, "models")
        or report.get("status") not in _MEASURED_STATUSES
        or report.get("mode") not in ("weights", "context")
        or report.get("model_verified") is not True
        or report.get("inference_executed") is not False
        or report.get("generated_tokens") != 0
        or report.get("cpu_fallback_selected") is not False
        or report.get("native_binding_version") is None
        or not isinstance(report.get("model_sha256"), str)
        or _SHA.fullmatch(report["model_sha256"]) is None
        or not isinstance(report.get("gpu_preflight"), dict)
        or report["gpu_preflight"].get("status") != "gpu_ready"
        or report["gpu_preflight"].get("processes")
    ):
        return result
    # Do not mistake a failed Python worker or unmeasured refusal for a
    # native capacity measurement.
    if report["status"] == "gpu_load_only_failed" and not (
        report.get("native_cuda_buffer_mib_reported") is not None
        or any(isinstance(x, dict) and x.get("kind") in ("model", "kv", "compute")
               and str(x.get("device", "")).startswith("CUDA")
               for x in (report.get("native_buffers") or []))
    ):
        return result

    root = Path(project_root)
    try:
        # Internal probe_model_load() returns a dict without the CLI's
        # presentation-only action. The strict local evidence validator
        # requires action=models; normalize BEFORE hashing and archiving.
        canonical_report = {"action": "models", **report}
        encoded = (json.dumps(canonical_report, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > 140_000:
            raise ValueError("Native probe evidence exceeds safe local size")
        digest = hashlib.sha256(encoded).hexdigest()
        base = root / ".ruth-build" / "profiles" / "evidence"
        base.mkdir(parents=True, exist_ok=True)
        if base.is_symlink():
            raise ValueError("Unsafe evidence directory")
        target = base / (report["model_sha256"] + "-" + digest + ".json")
        if target.is_symlink():
            raise ValueError("Unsafe evidence path")
        if target.exists():
            if not target.is_file() or target.read_bytes() != encoded:
                raise ValueError("Prior probe evidence conflicts with digest")
        else:
            temp_name = None
            try:
                with tempfile.NamedTemporaryFile(dir=base, prefix=".probe-",
                                                 suffix=".json", delete=False) as fp:
                    temp_name = fp.name
                    fp.write(encoded)
                    fp.flush()
                    os.fsync(fp.fileno())
                os.replace(temp_name, target)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)
        result["local_evidence_file"] = str(target)
        from .profile_registry import record_local_profile
        saved = record_local_profile(target, project_root=root)
        result["local_profile_update"] = saved
        result["status"] = saved["status"]
        if not sync_git:
            result["git_publication"] = {
                "status": "git_sync_disabled", "git_updated": False}
            return result
        from .profile_contribution import (
            build_proposal, sync_owner_profile_if_authenticated,
        )
        proposal = build_proposal(Path(saved["profile_file"]), project_root=root)
        publication = sync_owner_profile_if_authenticated(proposal)
        result["git_publication"] = publication
        result["runtime_profile_published"] = publication.get("status") in (
            "git_profile_updated_model_profiling",
            "git_profile_already_current",
        )
        return result
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        # The native probe result is unaffected. Local evidence, if present,
        # survives a failure to create or sync the sanitized Git proposal.
        result["status"] = "native_probe_recording_or_sync_failed"
        result["error"] = type(exc).__name__ + ": " + str(exc)[:230]
        return result


def recover_archived_probe(
    evidence_file: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    sync_git: bool = True,
) -> dict:
    """Replay the previously archived native measurement without CUDA.

    Accept only this project's content-addressed .ruth-build evidence path.
    In particular, do not accept arbitrary JSON on disk as a 'probe' or
    allow the filename to bypass the existing strict profile validators.
    """
    root = Path(project_root)
    archive = root / ".ruth-build" / "profiles" / "evidence"
    path = Path(evidence_file)
    if archive.is_symlink() or path.is_symlink():
        raise ValueError("Refuse symlink native evidence")
    if (path.resolve().parent != archive.resolve()
        or not path.is_file()
        or not 1 <= path.stat().st_size <= 140_000):
        raise ValueError("Recovery requires an existing local archived probe JSON")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    report = json.loads(raw)
    if (not isinstance(report, dict)
        or not isinstance(report.get("model_sha256"), str)
        or _SHA.fullmatch(report["model_sha256"]) is None
        or path.name != report["model_sha256"] + "-" + digest + ".json"
        or report.get("action") not in (None, "models")):
        raise ValueError("Archived probe filename/content digest or identity mismatch")
    return record_probe_evidence(report, project_root=root, sync_git=sync_git)
