"""Owner-only compare-and-swap publication into one canonical model profile.

Input is the already validated/privacy-stripped v1 proposal. GitHub is the
only source of truth; refresh on write conflicts and never upload raw evidence.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .model_profile_v2 import create_profile, merge_published_measurement, profile_path
from .model_catalog import load_catalog


def canonical_profile_commit(
    proposal: dict, *, run: Callable[..., Any],
    upstream: str = "yannericetchart-dev/ruth",
    branch: str = "model-profiling",
) -> dict:
    from .profile_contribution import (
        _validate_proposal_for_publication, validate_submitted_proposal)
    digest, _ = _validate_proposal_for_publication(proposal)
    observation = proposal["proposal"]
    # Run EXACTLY the same pinned-source and native-context validator as
    # historical Git submissions before touching authenticated GitHub.
    gguf_sha = (observation.get("model") or {}).get("gguf_sha256")
    if not isinstance(gguf_sha, str) or len(gguf_sha) != 64:
        raise ValueError("Missing pinned GGUF provenance")
    with TemporaryDirectory() as tmp:
        evidence_path = Path(tmp) / "profiles" / "submissions" / gguf_sha / (digest + ".json")
        evidence_path.parent.mkdir(parents=True)
        evidence_path.write_text(json.dumps(observation, sort_keys=True),
                                 encoding="utf-8")
        validate_submitted_proposal(evidence_path)

    if observation.get("schema") != "ruth-community-model-memory-proposal-v1":
        raise ValueError("Invalid sanitized model measurement schema")
    model = observation.get("model") or {}
    entry = next((e for e in load_catalog()["models"]
                  if all(e.get(k) == model.get(k) for k in
                         ("family", "model", "version"))), None)
    if entry is None:
        raise ValueError("Unpinned model cannot be published as canonical profile")
    path = profile_path(entry["family"], entry["model"], entry["version"]).relative_to(
        PROJECT_ROOT).as_posix()
    endpoint = f"repos/{upstream}/contents/{path}"
    for attempt in range(3):
        got = run(["gh", "api", endpoint + "?ref=" + branch],
                  capture_output=True, text=True, check=False, timeout=30)
        if got.returncode == 0:
            data = json.loads(got.stdout)
            if (not isinstance(data, dict)
                or data.get("type") != "file"
                or not isinstance(data.get("sha"), str)
                or len(data["sha"]) != 40):
                raise RuntimeError("Unexpected canonical GitHub profile response")
            old = base64.b64decode(data.get("content", "").replace("\n", ""),
                                   validate=True).decode("utf-8")
            canonical = json.loads(old)
            github_sha = data["sha"]
        elif "HTTP 404" in (got.stderr or ""):
            canonical = create_profile(entry)
            github_sha = None
        else:
            raise RuntimeError("GitHub canonical profile lookup failed")
        if canonical.get("model", {}).get("version") != entry["version"]:
            raise ValueError("Canonical model identity mismatch")
        updated, changed = merge_published_measurement(canonical, observation)
        if not changed and github_sha is not None:
            return {"status": "git_profile_already_current", "git_updated": False,
                    "repository": upstream, "repository_path": path,
                    "branch": branch, "review_required": False}
        raw = (json.dumps(updated, sort_keys=True, ensure_ascii=False,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > 2_500_000:
            raise ValueError("Canonical model profile size limit exceeded")
        request = [
            "api", "--method", "PUT", endpoint,
            "-f", f"message=profiles: merge verified model observations {entry['version']}",
            "-f", "content=" + base64.b64encode(raw).decode("ascii"),
            "-f", f"branch={branch}",
        ]
        if github_sha is not None:
            request.extend(["-f", "sha=" + github_sha])
        put = run(["gh", *request], capture_output=True,
                  text=True, check=False, timeout=30)
        if put.returncode == 0:
            response = json.loads(put.stdout)
            if ((response.get("content") or {}).get("path") != path
                or len((response.get("commit") or {}).get("sha", "")) != 40):
                raise RuntimeError("GitHub did not confirm canonical update")
            return {"status": "git_profile_updated_model_profiling",
                    "git_updated": True, "repository": upstream,
                    "repository_path": path, "branch": branch,
                    "commit_sha": response["commit"]["sha"], "review_required": False}
        if "HTTP 409" not in (put.stderr or "") and "HTTP 422" not in (put.stderr or ""):
            raise RuntimeError("GitHub canonical profile update rejected")
        # Concurrent publisher won: fetch again, remerge, and retry.
    raise RuntimeError("GitHub canonical profile CAS merge exceeded retry budget")


def sync_gguf_architecture(
    report: dict, *,
    run: Callable[..., Any] = subprocess.run,
    project_root = PROJECT_ROOT,
    branch: str = "model-profiling",
) -> dict:
    """Owner-only canonical CAS for a strictly pinned, read-only GGUF header.

    The report must originate from inspect_pinned_gguf and contain only
    deliberately allowlisted structural fields. No hardware fingerprints.
    """
    from .profile_contribution import _github_user, UPSTREAM, GitHubCLIMissing
    from .model_profile_v2 import profile_path, load_profile
    from .model_catalog import select_model
    if (report.get("status") != "gguf_header_geometry_observed_native_crosscheck_pending"
        or report.get("source_integrity_verified") is not True
        or report.get("architecture") != "gemma3"
        or report.get("weight_format") != "BF16"
        or report.get("tensor_values_read") is not False
        or report.get("cuda_initialized") is not False
        or report.get("model_weights_loaded") is not False
        or report.get("inference_executed") is not False):
        raise ValueError("Invalid header-only GGUF evidence")
    model = report.get("model_id") or {}
    if tuple(model.get(x) for x in ("family","model","version")) != (
        "gemma","gemma-3","gemma-3-1b-it"):
        raise ValueError("Only pinned Gemma3 header enrichment is allowed")
    chosen = select_model("production",family="gemma",model="gemma-3",
                          version="gemma-3-1b-it",format="gguf",
                          quantization="BF16",project_root=project_root)
    if (report.get("gguf_sha256") != chosen.expected_sha256
        or report.get("source_revision") != chosen.revision
        or report.get("source_repository") != chosen.repo_id
        or report.get("gguf_file_bytes") != chosen.expected_size_bytes):
        raise ValueError("GGUF header has unpinned source identity")
    fields = report.get("metadata_fields")
    counts = report.get("tensor_storage_counts")
    if (not isinstance(fields, dict)
        or not set(fields).issubset({
            "general.architecture","general.file_type","general.alignment",
            "gemma3.block_count","gemma3.embedding_length","gemma3.context_length",
            "gemma3.attention.head_count","gemma3.attention.head_count_kv",
            "gemma3.attention.key_length","gemma3.attention.value_length",
            "gemma3.attention.key_length_swa","gemma3.attention.value_length_swa",
            "gemma3.attention.sliding_window",
            "gemma3.attention.sliding_window_pattern",
        })
        or fields.get("general.architecture") != "gemma3"
        or fields.get("general.file_type") != 24
        or not isinstance(counts,dict)
        or len(counts) > 40
        or any(not isinstance(k,str) or len(k)>36
               or type(v) is not int or v<0 for k,v in counts.items())
        or sum(counts.values()) != report.get("tensor_count")
        or type(report.get("blocks")) is not int
        or report.get("blocks") != fields.get("gemma3.block_count")):
        raise ValueError("Unsafe or inconsistent GGUF header metadata")
    observed = {
        "status": "gguf_header_observed_native_crosscheck_pending",
        "gguf_sha256": chosen.expected_sha256,
        "gguf_version": report["gguf_version"],
        "tensor_count": report["tensor_count"],
        "tensor_storage_counts": counts,
        "metadata_fields": fields,
    }
    try:
        user = _github_user(run)
    except GitHubCLIMissing:
        return {"status":"git_cli_missing","git_updated":False,
                "retryable":True}
    if user.lower() != UPSTREAM.split("/")[0].lower():
        return {"status":"git_owner_review_required","git_updated":False,
                "review_required":True}
    root = Path(project_root)
    p = profile_path("gemma","gemma-3","gemma-3-1b-it",root)
    if not p.is_file():
        raise ValueError("Canonical pinned Gemma3 model missing locally")
    path = p.relative_to(root).as_posix()
    endpoint = "repos/" + UPSTREAM + "/contents/" + path
    for _ in range(3):
        get = run(["gh","api",endpoint+"?ref="+branch],
                  capture_output=True,text=True,check=False,timeout=30)
        if get.returncode != 0:
            raise RuntimeError("GitHub canonical Gemma3 profile unavailable")
        data = json.loads(get.stdout)
        if (data.get("type") != "file" or not isinstance(data.get("sha"),str)
            or len(data["sha"]) != 40):
            raise ValueError("Unrecognized canonical profile response")
        current = json.loads(base64.b64decode(
            data["content"].replace("\n",""),validate=True).decode("utf-8"))
        expected = load_profile("gemma","gemma-3","gemma-3-1b-it",
                                project_root=root)
        if (current.get("schema") != expected.get("schema")
            or current.get("model") != expected.get("model")
            or current["variants"]["BF16"]["gguf_sha256"] != chosen.expected_sha256):
            raise ValueError("Canonical model source mismatch")
        old_obs = (current.get("architecture") or {}).get("gguf_header_observation")
        if old_obs == observed:
            return {"status":"git_profile_already_current","git_updated":False,
                    "repository":UPSTREAM,"branch":branch,"repository_path":path}
        if old_obs is not None and old_obs != observed:
            raise ValueError("Conflicting native GGUF header evidence for pinned source")
        architecture = current.setdefault("architecture",{})
        architecture["gguf_header_observation"] = observed
        architecture["status"] = "gguf_header_observed_native_crosscheck_pending"
        # Only directly represented GGUF scalar facts are promoted.
        architecture["layer_count"] = report["blocks"]
        for field,meta_key in (
            ("swa_window","gemma3.attention.sliding_window"),
            ("kv_heads","gemma3.attention.head_count_kv"),
            ("key_head_dim","gemma3.attention.key_length"),
            ("value_head_dim","gemma3.attention.value_length"),
        ):
            val = fields.get(meta_key)
            if type(val) is int and val > 0:
                architecture[field] = val
        raw=(json.dumps(current,ensure_ascii=False,indent=2,
                        sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
        put=run(["gh","api","--method","PUT",endpoint,
                 "-f","message=profiles: record pinned Gemma3 GGUF header metadata",
                 "-f","content="+base64.b64encode(raw).decode("ascii"),
                 "-f","branch="+branch,
                 "-f","sha="+data["sha"]],
                capture_output=True,text=True,check=False,timeout=30)
        if put.returncode == 0:
            response=json.loads(put.stdout)
            if ((response.get("content") or {}).get("path") != path
                or len((response.get("commit") or {}).get("sha",""))!=40):
                raise RuntimeError("GitHub did not confirm architecture update")
            return {"status":"git_profile_updated_model_profiling",
                    "git_updated":True,"repository":UPSTREAM,
                    "branch":branch,"repository_path":path,
                    "commit_sha":response["commit"]["sha"]}
        if ("HTTP 409" not in (put.stderr or "")
            and "HTTP 422" not in (put.stderr or "")):
            raise RuntimeError("GitHub architecture publication rejected")
    raise RuntimeError("Concurrent Gemma3 canonical profile updates did not converge")
