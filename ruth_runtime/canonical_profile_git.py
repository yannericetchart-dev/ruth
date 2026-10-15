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
    matrix = report.get("matrix_tensor_elements_by_storage")
    warning_flags = report.get("metadata_warnings")
    declared = report.get("declared_file_type")
    if (not isinstance(matrix, dict)
        or not matrix or len(matrix) > 40
        or any(not isinstance(k,str) or len(k)>36
               or type(v) is not int or not 0 <= v <= 10**15
               for k,v in matrix.items())
        or matrix.get("BF16", 0) <= (
            sum(matrix.values()) - matrix.get("BF16", 0))
        or not isinstance(warning_flags,list)
        or len(warning_flags) > 3
        or any(x not in (
            "general.file_type_missing",
            "general.file_type_differs_from_BF16_tensors",
            "matrix_tensor_storage_is_mixed") for x in warning_flags)
        or (declared is not None and (type(declared) is not int
                                      or not 0 <= declared < 2**32))
        or declared != fields.get("general.file_type")
        or ((declared != 24) != any(
            x in warning_flags for x in (
                "general.file_type_missing",
                "general.file_type_differs_from_BF16_tensors")))
        or ("matrix_tensor_storage_is_mixed" in warning_flags) != (len(matrix) > 1)):
        raise ValueError("Unsafe or inconsistent GGUF tensor BF16 evidence")
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
        or not isinstance(counts,dict)
        or len(counts) > 40
        or any(not isinstance(k,str) or len(k)>36
               or type(v) is not int or v<0 for k,v in counts.items())
        or sum(counts.values()) != report.get("tensor_count")
        or type(report.get("blocks")) is not int
        or report.get("blocks") != fields.get("gemma3.block_count")):
        raise ValueError("Unsafe or inconsistent GGUF header metadata")
    # Validate whitelisted per-layer weight descriptors independently from
    # the KV cache, which stores activations and may use a different dtype.
    tensor_entries = report.get("attention_weight_tensors")
    coverage = report.get("attention_projection_coverage")
    if tensor_entries is not None or coverage is not None:
        from .gguf_header_inspection import ATTENTION_ROLES, ATTENTION_PROJECTORS
        from collections import Counter
        layers = report.get("blocks")
        if (not isinstance(tensor_entries, list)
            or not isinstance(coverage, dict)
            or set(coverage) != set(ATTENTION_PROJECTORS)
            or type(layers) is not int or layers != 26
            or len(tensor_entries) > layers * len(ATTENTION_ROLES)
            or report.get("attention_dtypes_are_kv_cache_dtypes") is not False):
            raise ValueError("Unsafe attention tensor role report")
        unique = set()
        for item in tensor_entries:
            if (not isinstance(item, dict)
                or set(item) != {"layer", "role", "dtype", "shape"}
                or type(item["layer"]) is not int
                or not 0 <= item["layer"] < layers
                or item["role"] not in ATTENTION_ROLES
                or item["dtype"] not in ("BF16","F16","F32")
                or not isinstance(item["shape"], list)
                or not 1 <= len(item["shape"]) <= 4
                or any(type(dim) is not int or not 1 <= dim <= 1_000_000
                       for dim in item["shape"])):
                raise ValueError("Unexpected attention tensor role descriptor")
            key = (item["layer"], item["role"])
            if key in unique:
                raise ValueError("Duplicate attention tensor role")
            unique.add(key)
        for role in ATTENTION_PROJECTORS:
            row = coverage[role]
            selected = [x for x in tensor_entries if x["role"] == role]
            types = dict(sorted(Counter(x["dtype"] for x in selected).items()))
            if (not isinstance(row,dict)
                or set(row) != {"layers_present", "dtype_counts", "complete"}
                or row["layers_present"] != len(selected)
                or row["dtype_counts"] != types
                or type(row["complete"]) is not bool
                or row["complete"] != (len(selected) == layers
                    and {x["layer"] for x in selected} == set(range(layers)))):
                raise ValueError("Inconsistent GGUF per-layer projection coverage")
    else:
        tensor_entries = None
        coverage = None
    observed = {
        "status": "gguf_header_observed_native_crosscheck_pending",
        "gguf_sha256": chosen.expected_sha256,
        "gguf_version": report["gguf_version"],
        "tensor_count": report["tensor_count"],
        "tensor_storage_counts": counts,
        "matrix_tensor_elements_by_storage": matrix,
        "declared_file_type": declared,
        "metadata_warnings": warning_flags,
        "metadata_fields": fields,
    }
    if tensor_entries is not None:
        observed["attention_weight_tensors"] = tensor_entries
        observed["attention_projection_coverage"] = coverage
        observed["attention_dtypes_are_kv_cache_dtypes"] = False
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
        if old_obs is not None and any(
            observed.get(field) != value for field, value in old_obs.items()
        ):
            # Existing pinned observations must never be rewritten by a
            # re-inspection: optional role data may only ENRICH them.
            raise ValueError("Conflicting native GGUF header evidence for pinned source")
        # Recover deterministic attention partition from the pinned model
        # GGUF header AND the pinned llama.cpp Gemma3 SWA fallback rule.
        from .gemma3_kv_profile import derived_gemma3_geometry
        source_input = {
            "model": current["model"], "variants": current["variants"],
            "backend_policies": current["backend_policies"],
            "architecture": {"gguf_header_observation": observed},
        }
        source_geom = derived_gemma3_geometry(source_input)
        old_source = (current.get("architecture") or {}).get("source_derived_geometry")
        if old_source is not None and old_source != source_geom:
            raise ValueError("Conflicting pinned source-derived Gemma 3 geometry")
        if old_obs == observed and old_source == source_geom:
            return {"status":"git_profile_already_current","git_updated":False,
                    "repository":UPSTREAM,"branch":branch,"repository_path":path}
        architecture = current.setdefault("architecture",{})
        architecture["gguf_header_observation"] = observed
        architecture["source_derived_geometry"] = source_geom
        architecture["global_layers"] = source_geom["global_layers"]
        architecture["swa_layers"] = source_geom["swa_layers"]
        architecture["status"] = "gguf_header_and_pinned_backend_source_derived_native_context_unverified"
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
