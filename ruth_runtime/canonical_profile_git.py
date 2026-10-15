"""Owner-only compare-and-swap publication into one canonical model profile.

Input is the already validated/privacy-stripped v1 proposal. GitHub is the
only source of truth; refresh on write conflicts and never upload raw evidence.
"""
from __future__ import annotations

import base64
import json
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
