"""Publish privacy-minimized memory profiles via GitHub identity gates.

Authenticated repository owner writes immutable measurements DIRECTLY to
model-profiling; all other accounts are PR-only and require owner's review.
Never reads or prints credentials or invokes a shell.
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
from .model_catalog import load_catalog
from .hardware_eligibility import load_architecture_policy
from .install_plan import LLAMA_CPP_SUBMODULE_COMMIT
from .llama_kv_profile import calculate_llama_kv

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
    context_data = data.get("context_profiles", {})
    if not isinstance(context_data, dict) or len(context_data) > 128:
        raise ValueError("Malformed native context memory profiles")
    context_fields = (
        "n_ctx_requested", "kv_size_native", "kv_padding",
        "flash_attention", "kv_type_k", "kv_type_v",
        "gpu_kv_buffer_mib", "gpu_compute_buffer_mib",
        "backend", "verified_native_allocations_only", "inference_verified",
        "kv_backend_commit", "native_config", "kv_source_derivation",
    )
    public_contexts = []
    for context_key, ctx in context_data.items():
        if (not isinstance(context_key, str) or len(context_key) > 384
            or not isinstance(ctx, dict)):
            raise ValueError("Invalid context memory entry")
        record = {key: ctx.get(key) for key in context_fields}
        if (record["backend"] != "llama.cpp"
            or record["verified_native_allocations_only"] is not True
            or record["inference_verified"] is not False
            or type(record["n_ctx_requested"]) is not int
            or not 1 <= record["n_ctx_requested"] <= 131072):
            raise ValueError("Unverified CUDA KV/compute profile")
        # Exact runtime identity is in the local context key and remains
        # separate from the quantization/filename namespace.
        matching = [row for key, row in rows.items()
                    if context_key.startswith(key + "/n_ctx=")
                    and context_key.split("/n_ctx=")[1].startswith(
                        str(record["n_ctx_requested"]) + "/")]
        if len(matching) != 1:
            raise ValueError("Context runtime identity ambiguous or mismatched")
        runtime = matching[0]
        record["architecture"] = runtime.get("architecture")
        record["cuda_backend"] = runtime.get("cuda_backend")
        record["driver_version"] = runtime.get("driver_version")
        record["llama_cpp_binding"] = runtime.get("llama_cpp_binding")
        public_contexts.append(record)
    if public_contexts:
        proposal["context_profiles"] = sorted(
            public_contexts, key=lambda q: (q["architecture"], q["driver_version"],
                                           q["cuda_backend"], q["n_ctx_requested"]))
    observed = data.get("minimal_context_inference", [])
    if not isinstance(observed, list) or len(observed) > 64:
        raise ValueError("Malformed minimal-context evidence")
    if observed:
        safe_obs = []
        fields = ("verified", "n_ctx", "n_batch", "n_ubatch",
                  "generated_tokens", "full_gpu_offload_layers",
                  "architecture", "driver_version", "cuda_backend",
                  "llama_cpp_binding", "flash_attention",
                  "kv_type_k", "kv_type_v", "production_context_qualified")
        for entry in observed:
            if not isinstance(entry, dict):
                raise ValueError("Invalid native inference entry")
            record = {k: entry.get(k) for k in fields}
            if (record["verified"] is not True
                or record["production_context_qualified"] is not False
                or type(record["n_ctx"]) is not int or record["n_ctx"] < 1
                or type(record["generated_tokens"]) is not int or record["generated_tokens"] < 1
                or type(record["full_gpu_offload_layers"]) is not int
                or record["full_gpu_offload_layers"] < 1):
                raise ValueError("Minimal inference proof incomplete or overstated")
            safe_obs.append(record)
        proposal["minimal_context_inference"] = sorted(
            safe_obs, key=lambda q:(q["architecture"], q["driver_version"],
                                    q["cuda_backend"], q["n_ctx"]))
    raw = json.dumps(proposal, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return {"proposal": proposal, "content_digest": digest,
            "repository_path": f"profiles/submissions/{sha}/{digest}.json"}



def validate_submitted_proposal(
    file: Path, *, project_root: Path = PROJECT_ROOT,
) -> dict:
    """CI-side, no network, validate proposal and owner-direct measurement files.

    This establishes only schema/provenance consistency, NOT that a stranger's
    physical measurements are trustworthy. Human review is still mandatory.
    """
    path = Path(file)
    parts = path.parts
    if (len(parts) < 4 or parts[-4] != "profiles"
        or parts[-3] not in ("submissions", "measurements")
        or not re.fullmatch("[0-9a-f]{64}", parts[-2])
        or not re.fullmatch("[0-9a-f]{64}\.json", parts[-1])):
        raise ValueError("Invalid submission path")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 200_000:
        raise ValueError("Unsafe or oversized submission")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
        or set(payload) - {"schema", "model", "runtime_measurements",
                          "verified_inference", "production_qualified",
                          "review_required", "measurements_pending",
                          "minimal_context_inference", "context_profiles"}
        or not {"schema", "model", "runtime_measurements",
                "verified_inference", "production_qualified",
                "review_required", "measurements_pending"}.issubset(payload)
        or payload["schema"] != "ruth-community-model-memory-proposal-v1"
        or payload["verified_inference"] is not False
        or payload["production_qualified"] is not False
        or payload["review_required"] is not True):
        raise ValueError("Unsupported or overclaimed submission")
    model = payload["model"]
    if not isinstance(model, dict) or model.get("gguf_sha256") != parts[-2]:
        raise ValueError("Submitted model SHA mismatch")
    catalog = load_catalog(project_root)
    pinned = False
    for item in catalog["models"]:
        for v in item["variants"]:
            if (model.get("gguf_sha256") == v["expected_sha256"]
                and model.get("quantization") == v["quantization"]
                and model.get("source_repository") == v["repo_id"]
                and model.get("source_revision") == v["revision"]
                and model.get("version") == item["version"]
                and model.get("family") == item["family"]
                and model.get("model") == item["model"]
                and model.get("format") == item["format"]
                and model.get("license") == item["license"]
                and model.get("gguf_serialized_bytes") == v["expected_size_bytes"]):
                pinned = True
    if not pinned:
        raise ValueError("Unregistered GGUF source")
    rows = payload["runtime_measurements"]
    if (not isinstance(rows, list) or len(rows) > 128 or
        payload["measurements_pending"] is not (len(rows) == 0)):
        raise ValueError("Invalid runtime measurement list")
    architectures = load_architecture_policy(project_root)["architectures"]
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid runtime row")
        if not set(row).issubset({
            "architecture", "compute_capability", "tensor_core_generation",
            "cuda_backend", "cuda_driver_max", "cuda_toolkit", "driver_version",
            "llama_cpp_binding", "cuda_init_observed_nvml_delta_mib_max",
            "driver_reserved_mib_max", "weights_buffer_mib_max",
            "kv_buffer_mib", "compute_buffer_mib", "n_ctx",
            "flash_attention", "kv_type_k", "kv_type_v",
        }):
            raise ValueError("Unknown or private runtime field")
        matching = [a for a in architectures
                    if a["id"] == row.get("architecture")
                    and row.get("compute_capability") in a["compute_capabilities"]]
        if len(matching) != 1 or row.get("tensor_core_generation") != matching[0]["tensor_core_generation"]:
            raise ValueError("Architecture/compute capability mismatch")
        for n in ("weights_buffer_mib_max", "cuda_init_observed_nvml_delta_mib_max",
                  "driver_reserved_mib_max", "kv_buffer_mib", "compute_buffer_mib"):
            value = row.get(n)
            if value is not None and (type(value) not in (int, float) or
                                       not 0 <= value < 1_000_000):
                raise ValueError("Invalid CUDA buffer measurement")
    public_contexts = payload.get("context_profiles", [])
    if not isinstance(public_contexts, list) or len(public_contexts) > 128:
        raise ValueError("Invalid published CUDA context profiles")
    if public_contexts:
        context_fields = {
            "n_ctx_requested", "kv_size_native", "kv_padding",
            "flash_attention", "kv_type_k", "kv_type_v",
            "gpu_kv_buffer_mib", "gpu_compute_buffer_mib", "backend",
            "verified_native_allocations_only", "inference_verified",
            "architecture", "cuda_backend", "driver_version",
            "llama_cpp_binding",
            "kv_backend_commit", "native_config", "kv_source_derivation",
        }
        for ctx in public_contexts:
            if (not isinstance(ctx, dict) or set(ctx) != context_fields
                or ctx["backend"] != "llama.cpp"
                or ctx["verified_native_allocations_only"] is not True
                or ctx["inference_verified"] is not False
                or type(ctx["n_ctx_requested"]) is not int
                or not 1 <= ctx["n_ctx_requested"] <= 131072
                or not any(row.get("architecture") == ctx["architecture"]
                           and row.get("driver_version") == ctx["driver_version"]
                           and row.get("cuda_backend") == ctx["cuda_backend"]
                           and row.get("llama_cpp_binding") == ctx["llama_cpp_binding"]
                           for row in rows)):
                raise ValueError("Context claims not linked to exact native runtime")
            # Old published measurements are valid legacy observations.
            # New option-aware context profiles MUST carry an immutable
            # source revision and a re-computable calculation.
            backend_commit = ctx.get("kv_backend_commit")
            native_cfg = ctx.get("native_config")
            derived = ctx.get("kv_source_derivation")
            if backend_commit is not None or native_cfg is not None or derived is not None:
                if (backend_commit != LLAMA_CPP_SUBMODULE_COMMIT
                    or not isinstance(native_cfg, dict)
                    or not isinstance(derived, dict)):
                    raise ValueError("Option-aware KV profile has no exact backend provenance")
                needed = {"n_batch_requested", "n_ubatch_requested", "swa_full",
                          "flash_attention_requested", "flash_attention_effective",
                          "n_ctx_effective_native", "n_ctx_seq_effective_native",
                          "n_batch_effective_native", "n_ubatch_effective_native",
                          "global_cache_cells_native", "swa_cache_cells_native",
                          "global_key_logical_mib_native", "global_value_logical_mib_native",
                          "swa_key_logical_mib_native", "swa_value_logical_mib_native"}
                if set(native_cfg) != needed or native_cfg["flash_attention_effective"] is not None:
                    raise ValueError("Unknown native KV diagnostics or unsupported effective FA claim")
                if (native_cfg["flash_attention_requested"] != ctx["flash_attention"]
                    or not isinstance(native_cfg["swa_full"], bool)
                    or any(type(native_cfg[k]) is not int for k in
                           ("n_batch_requested", "n_ubatch_requested"))):
                    raise ValueError("Inconsistent KV options")
                computed = calculate_llama_kv(
                    project_root=project_root, n_ctx=ctx["n_ctx_requested"],
                    n_batch=native_cfg["n_batch_requested"],
                    n_ubatch=native_cfg["n_ubatch_requested"],
                    swa_full=native_cfg["swa_full"],
                    flash_attention=native_cfg["flash_attention_requested"],
                    type_k=ctx["kv_type_k"], type_v=ctx["kv_type_v"],
                )
                expected = {
                    "n_ctx_effective_derived": computed["configuration"]["n_ctx_effective_derived"],
                    "n_ubatch_effective_derived": computed["configuration"]["n_ubatch_effective_derived"],
                    "global_cells_derived": computed["global_cache"]["cells"],
                    "swa_cells_derived": computed["swa_cache"]["cells"],
                    "key_logical_bytes_derived": computed["key_logical_bytes_derived"],
                    "value_logical_bytes_derived": computed["value_logical_bytes_derived"],
                    "swa_full_off_saves_bytes_derived": computed["swa_full_off_saves_bytes_derived"],
                    "swa_off_first_saving_at_effective_ctx":
                        computed["swa_off_first_saving_at_effective_ctx"],
                    "model_geometry_verification_pending": True,
                }
                if derived != expected:
                    raise ValueError("Published source-derived KV sizes differ from pinned backend")
                for actual, planned in (
                    ("n_ctx_effective_native", expected["n_ctx_effective_derived"]),
                    ("global_cache_cells_native", expected["global_cells_derived"]),
                    ("swa_cache_cells_native", expected["swa_cells_derived"]),
                ):
                    observed = native_cfg[actual]
                    if observed is not None and (type(observed) is not int or observed != planned):
                        raise ValueError("Native KV evidence inconsistent with backend source")
                for observed in ("global_key_logical_mib_native",
                                 "global_value_logical_mib_native",
                                 "swa_key_logical_mib_native",
                                 "swa_value_logical_mib_native"):
                    v = native_cfg[observed]
                    if v is not None and (type(v) not in (float, int) or not 0 <= v < 1_000_000):
                        raise ValueError("Invalid native logical KV quantity")
            for value in (ctx["gpu_kv_buffer_mib"], ctx["gpu_compute_buffer_mib"]):
                if value is not None and (
                    type(value) not in (float, int)
                    or not 0 <= value <= 1000000):
                    raise ValueError("Invalid native buffer allocation")
    minimal = payload.get("minimal_context_inference", [])
    if not isinstance(minimal, list) or len(minimal) > 64:
        raise ValueError("Invalid minimal CUDA inference records")
    if minimal:
        allowed_fields = {
            "verified", "n_ctx", "n_batch", "n_ubatch",
            "generated_tokens", "full_gpu_offload_layers",
            "architecture", "driver_version", "cuda_backend",
            "llama_cpp_binding", "flash_attention",
            "kv_type_k", "kv_type_v", "production_context_qualified",
        }
        for entry in minimal:
            if (not isinstance(entry, dict)
                or set(entry) != allowed_fields
                or entry["verified"] is not True
                or entry["production_context_qualified"] is not False
                or type(entry["n_ctx"]) is not int or entry["n_ctx"] < 1
                or type(entry["generated_tokens"]) is not int or entry["generated_tokens"] < 1
                or type(entry["full_gpu_offload_layers"]) is not int
                or entry["full_gpu_offload_layers"] < 1
                or not any(
                    row.get("architecture") == entry["architecture"]
                    and row.get("driver_version") == entry["driver_version"]
                    and row.get("cuda_backend") == entry["cuda_backend"]
                    and row.get("llama_cpp_binding") == entry["llama_cpp_binding"]
                    for row in rows)):
                raise ValueError("Unmatched or unverified GPU inference record")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if parts[-1] != digest + ".json":
        raise ValueError("Submission digest invalid or content altered")
    return {"status": "submitted_profile_schema_verified_review_required",
            "model_sha256": model["gguf_sha256"], "observations": len(rows),
            "production_qualified": False}


class GitHubCLIMissing(RuntimeError):
    """The external GitHub CLI is not installed on the host."""


def _gh(args: list[str], run: Callable[..., Any]) -> str:
    try:
        result = run(["gh", *args], capture_output=True, text=True,
                     timeout=30, check=False)
    except FileNotFoundError as exc:
        raise GitHubCLIMissing("GitHub CLI 'gh' not found on PATH") from exc
    if result.returncode != 0:
        raise RuntimeError("Authenticated GitHub CLI operation failed: " +
                           " ".join(args[:2]))
    return result.stdout.strip()


def _validate_proposal_for_publication(proposal: dict) -> tuple[str, str]:
    """Reject mutations before any GitHub read or write."""
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
    return digest, path


def _github_user(run: Callable[..., Any]) -> str:
    user = _gh(["api", "user", "--jq", ".login"], run)
    if not _LOGIN.fullmatch(user):
        raise ValueError("Unexpected authenticated GitHub login")
    return user


def _owner_direct_commit(
    proposal: dict, *, run: Callable[..., Any],
) -> dict:
    """Append immutable model evidence DIRECTLY to model-profiling.

    Use GitHub Contents API with branch explicitly set. Never overwrite existing
    paths, and do not write to release/main. A same-content retry is idempotent.
    """
    digest, submission_path = _validate_proposal_for_publication(proposal)
    path = submission_path.replace("profiles/submissions/", "profiles/measurements/", 1)
    endpoint = f"repos/{UPSTREAM}/contents/{path}"
    public_content = json.dumps(proposal["proposal"], sort_keys=True, indent=2) + "\n"
    # GitHub returns HTTP 404 for a missing path. Other failures fail closed.
    try:
        response = run(
            ["gh", "api", endpoint + "?ref=" + BASE_BRANCH],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except FileNotFoundError as exc:
        raise GitHubCLIMissing("GitHub CLI 'gh' not found on PATH") from exc
    if response.returncode == 0:
        data = json.loads(response.stdout)
        if not isinstance(data, dict) or data.get("type") != "file":
            raise RuntimeError("Unexpected GitHub profile content response")
        old = base64.b64decode(data.get("content", "").replace("\n", ""),
                               validate=True).decode("utf-8")
        if json.loads(old) != proposal["proposal"]:
            raise ValueError("Immutable profile path already has different content")
        return {
            "status": "git_profile_already_current", "git_updated": False,
            "branch": BASE_BRANCH, "repository": UPSTREAM,
            "repository_path": path, "review_required": False,
        }
    stderr = response.stderr or ""
    if not re.search(r"HTTP 404\b", stderr):
        raise RuntimeError("GitHub profile lookup failed (not an absent path)")
    content = base64.b64encode(public_content.encode("utf-8")).decode("ascii")
    result = _gh([
        "api", "--method", "PUT", endpoint,
        "-f", f"message=profiles: record measured CUDA profile {digest[:16]}",
        "-f", f"content={content}",
        "-f", f"branch={BASE_BRANCH}",
    ], run)
    confirmed = json.loads(result)
    if (not isinstance(confirmed, dict)
        or (confirmed.get("content") or {}).get("path") != path
        or not re.fullmatch("[0-9a-f]{40}",
                            (confirmed.get("commit") or {}).get("sha", ""))):
        raise RuntimeError("GitHub did not verify branch file commit")
    return {
        "status": "git_profile_updated_model_profiling",
        "git_updated": True, "review_required": False,
        "branch": BASE_BRANCH, "repository": UPSTREAM,
        "repository_path": path,
        "commit_sha": confirmed["commit"]["sha"],
    }


def sync_owner_profile_if_authenticated(
    proposal: dict, *, run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Automatic owner-only commit, no extra consent flag; others stay local."""
    try:
        _validate_proposal_for_publication(proposal)
        login = _github_user(run)
        if login.lower() != UPSTREAM.split("/")[0].lower():
            return {
                "status": "git_owner_review_required",
                "git_updated": False, "review_required": True,
                "next_step": "An explicit --sync-git submits a fork PR for owner approval",
            }
        return _owner_direct_commit(proposal, run=run)
    except GitHubCLIMissing:
        return {
            "status": "git_cli_missing", "git_updated": False,
            "retryable": True, "review_required": None,
            "remediation": "Install gh, authenticate, then retry --sync-git with saved profile",
        }


def publish_proposal_pr(
    proposal: dict, *, permit_external_publication: bool = False,
    run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Legacy explicit publication API: owner direct, contributor PR."""
    if not permit_external_publication:
        return {"status": "git_publication_not_authorized", "git_updated": False}
    return sync_profile_to_git(proposal, run=run)


def sync_profile_to_git(
    proposal: dict, *, run: Callable[..., Any] = subprocess.run,
) -> dict:
    """Authenticated owner: direct branch; outside contributor: PR, never push."""
    try:
        _validate_proposal_for_publication(proposal)
        user = _github_user(run)
        if user.lower() == UPSTREAM.split("/")[0].lower():
            return _owner_direct_commit(proposal, run=run)
        return _publish_external_pr(proposal, user=user, run=run)
    except GitHubCLIMissing:
        return {
            "status": "git_cli_missing", "git_updated": False,
            "review_required": None, "retryable": True,
            "remediation": "Install GitHub CLI (gh), run gh auth login and retry with saved profile",
        }


def _publish_external_pr(
    proposal: dict, *, user: str, run: Callable[..., Any],
) -> dict:
    """Only nonowners use forks, with owner's review required at merge."""
    digest, path = _validate_proposal_for_publication(proposal)
    # gh repo fork OWNER/REPO --clone=false leaves local git remotes alone.
    try:
        existing = json.loads(_gh(["api", f"repos/{user}/ruth"], run))
    except RuntimeError:
        existing = None
    if existing is None:
        _gh(["repo", "fork", UPSTREAM, "--clone=false"], run)
    elif existing.get("fork") is not True or (
        existing.get("parent") or {}).get("full_name", "").lower() != UPSTREAM.lower():
        raise ValueError("A repository named ruth already exists but is not Ruth's fork")
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
        return {"status": "git_branch_exists_review_existing_pr",
                "git_updated": False, "branch": branch, "review_required": True}
    public_content = json.dumps(proposal["proposal"], sort_keys=True, indent=2) + "\n"
    encoded = base64.b64encode(public_content.encode("utf-8")).decode("ascii")
    _gh(["api", "--method", "PUT", f"repos/{fork}/contents/{path}",
         "-f", f"message=profiles: submit immutable measured CUDA report {digest[:16]}",
         "-f", f"content={encoded}", "-f", f"branch={branch}"], run)
    title = f"Profile proposal: pinned GGUF CUDA measurements {digest[:12]}"
    body = ("Automated privacy-minimized measurement proposal. "
            "Only repository owner approval can integrate this contribution. "
            "Does not claim verified production inference.")
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

def main(argv: list[str] | None = None) -> int:
    """Offline validation entrypoint for untrusted community pull requests."""
    import argparse
    p = argparse.ArgumentParser(description="Validate pinned profile submission (offline)")
    p.add_argument("--validate", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        result = validate_submitted_proposal(args.validate)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "submitted_profile_rejected",
                          "error": type(exc).__name__ + ": " + str(exc)[:200]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
