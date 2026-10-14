"""Privacy-minimized model profile proposals and opt-in GitHub fork PR.

Never commits or pushes into the upstream repository from arbitrary users.
Requires authenticated GitHub CLI, exact explicit operator approval and a
reviewable fork PR; no token read/printed, no shell invocation.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .profile_registry import SCHEMA

UPSTREAM = "yannericetchart-dev/ruth"
BASE_BRANCH = "model-profiling"
_SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_LOGIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}$")


def build_proposal(profile_file: Path, *, project_root: Path = PROJECT_ROOT) -> dict:
    root = Path(project_root)
    path = Path(profile_file)
    expected = root / ".ruth-build/profiles/models"
    if path.is_symlink() or path.resolve().parent != expected.resolve():
        raise ValueError("Only a local Ruth memory profile may be submitted")
    if not path.is_file() or path.stat().st_size > 200_000:
        raise ValueError("Missing/oversized local profile")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("Unrecognized local profile")
    model = data.get("model") or {}
    sha = model.get("gguf_sha256")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("Invalid pinned GGUF")
    rows = data.get("runtime_measurements")
    if not isinstance(rows, dict) or (not rows and data.get("source_verified") is not True):
        raise ValueError("Missing source verification or measured CUDA data")
    # Allowlisted fields only. Do NOT upload GPU UUID, BusID, paths, process
    # names, host username, machine identity, raw prompts or native stderr.
    clean = []
    allowed = (
        "architecture", "compute_capability", "tensor_core_generation",
        "cuda_backend", "cuda_driver_max", "cuda_toolkit", "driver_version",
        "llama_cpp_binding", "cuda_init_observed_nvml_delta_mib_max",
        "driver_reserved_mib_max", "weights_buffer_mib_max",
        "kv_buffer_mib", "compute_buffer_mib", "n_ctx",
        "flash_attention", "kv_type_k", "kv_type_v",
    )
    for row in rows.values():
        if not isinstance(row, dict):
            raise ValueError("Malformed runtime row")
        safe = {name: row.get(name) for name in allowed}
        if any(not isinstance(safe[name], str) or not _SEGMENT.fullmatch(safe[name])
               for name in ("architecture", "compute_capability", "cuda_backend",
                            "cuda_driver_max", "driver_version", "llama_cpp_binding")):
            raise ValueError("Invalid profile identifiers")
        clean.append(safe)
    clean.sort(key=lambda x: (x["architecture"], x["cuda_backend"],
                              x["cuda_driver_max"], x["driver_version"]))
    safe_model = {k: model.get(k) for k in (
        "family", "model", "version", "format", "quantization",
        "gguf_sha256", "source_repository", "source_revision",
        "gguf_serialized_bytes", "license",
    )}
    proposal = {
        "schema": "ruth-community-model-memory-proposal-v1",
        "model": safe_model,
        "runtime_measurements": clean,
        "verified_inference": False,
        "production_qualified": False,
        "review_required": True,
        "measurements_pending": not bool(clean),
    }
    raw = json.dumps(proposal, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return {"proposal": proposal, "content_digest": digest,
            "repository_path": f"profiles/submissions/{sha}/{digest}.json"}


def _gh(args: list[str], run: Callable[..., Any]) -> str:
    result = run(["gh", *args], capture_output=True, text=True,
                 timeout=30, check=False)
    if result.returncode != 0:
        raise RuntimeError("Authenticated GitHub CLI operation failed: " +
                           " ".join(args[:2]))
    return result.stdout.strip()


def publish_proposal_pr(
    proposal: dict, *, permit_external_publication: bool = False,
    run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Push one immutable allowlisted JSON in user's fork and open upstream PR.

    User must opt in to publishing machine measurement summaries; requires
    Github CLI authentication with permission to create fork/PR.
    """
    if not permit_external_publication:
        return {"status": "git_publication_not_authorized", "git_updated": False}
    digest = proposal.get("content_digest")
    path = proposal.get("repository_path")
    if (not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest)
        or not isinstance(path, str)
        or not re.fullmatch(r"profiles/submissions/[0-9a-f]{64}/[0-9a-f]{64}\.json", path)
        or not isinstance(proposal.get("proposal"), dict)):
        raise ValueError("Invalid immutable proposal identity")
    canonical = json.dumps(proposal["proposal"], sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest:
        raise ValueError("Proposal digest changed")
    user = _gh(["api", "user", "--jq", ".login"], run)
    if not _LOGIN.fullmatch(user):
        raise ValueError("Unexpected GitHub login")
    _gh(["repo", "fork", UPSTREAM, "--clone=false", "--remote=false"], run)
    fork = f"{user}/ruth"
    head_json = _gh(["api", f"repos/{fork}/git/ref/heads/{BASE_BRANCH}"], run)
    head = json.loads(head_json)["object"]["sha"]
    if not re.fullmatch("[0-9a-f]{40}", head):
        raise ValueError("Cannot verify fork base commit")
    branch = f"ruth-profile-{digest[:16]}"
    try:
        _gh(["api", "--method", "POST", f"repos/{fork}/git/refs",
             "-f", f"ref=refs/heads/{branch}", "-f", f"sha={head}"], run)
    except RuntimeError:
        # Never overwrite an existing branch; return a clear manual state.
        return {"status": "git_branch_exists_review_existing_pr",
                "git_updated": False, "branch": branch}
    public_content = json.dumps(proposal["proposal"], sort_keys=True, indent=2) + "\n"
    encoded = base64.b64encode(public_content.encode("utf-8")).decode("ascii")
    _gh(["api", "--method", "PUT", f"repos/{fork}/contents/{path}",
         "-f", f"message=profiles: submit immutable measured CUDA report {digest[:16]}",
         "-f", f"content={encoded}", "-f", f"branch={branch}"], run)
    title = f"Profile proposal: pinned GGUF CUDA measurements {digest[:12]}"
    body = ("Automated, privacy-minimized **measurement proposal**. "
            "Requires schema/provenance validation and human review; "
            "no inference/production compatibility claims. "
            "No machine identifiers, prompts or credentials included.")
    response = _gh(["api", "--method", "POST", f"repos/{UPSTREAM}/pulls",
                    "-f", f"title={title}", "-f", f"head={user}:{branch}",
                    "-f", f"base={BASE_BRANCH}", "-f", f"body={body}"], run)
    pr = json.loads(response)
    url = pr.get("html_url")
    if not isinstance(url, str) or not url.startswith(
        f"https://github.com/{UPSTREAM}/pull/"):
        raise RuntimeError("GitHub did not confirm expected upstream PR")
    return {"status": "git_profile_pull_request_opened",
            "git_updated": False, "review_required": True,
            "pull_request_url": url, "branch": branch, "repository_path": path}
