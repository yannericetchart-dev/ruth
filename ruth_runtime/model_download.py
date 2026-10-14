"""Explicit, hash-pinned, resumable GGUF download from Hugging Face.

No external Python dependencies, implicit downloads, license acceptance,
inference, or changes to the Ruth .venv. A complete, verified local model
is promoted atomically; partial bytes can resume with strict HTTP 206.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable
from urllib.error import URLError, HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .dependencies import PROJECT_ROOT
from .model_catalog import (
    ModelChoice, REPO_ID, GIT_SHA, SHA256, resolve_model_metadata,
)

MODEL_ROOT = Path("models")
BUFFER = 1024 * 1024
SPARE_BYTES = 128 * 1024 * 1024
# Exact HTTPS hostnames, NOT wildcard trust. Hugging Face documents the
# active endpoints here (model download / firewall allowlist):
# https://huggingface.co/docs/hub/models-downloading
# Retain older known HF endpoints for already-mirrored GGUF files. Review
# and explicitly pin new hostnames if upstream infrastructure changes.
ALLOWED_DOWNLOAD_HOSTS = frozenset({
    "huggingface.co",
    "cas-server.xethub.hf.co",
    "cas-server.xethub-eu.hf.co",
    "transfer.xethub.hf.co",
    "transfer.xethub-eu.hf.co",
    "us.aws.cdn.hf.co",
    "us.gcp.cdn.hf.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
    "cdn-lfs.hf.co",
    "cdn-lfs.huggingface.co",
    "cas-bridge.xethub.hf.co",
})
RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def _validate_hf_download_url(url: str, *, origin: bool = False) -> str:
    """Accept only exact documented HF HTTPS hosts; never expose signed URLs.

    The hostname alone may be included in a diagnostic, not a URL with
    potentially sensitive Xet/LFS query parameters or bearer signatures.
    """
    parsed = urlsplit(url)
    host = parsed.hostname
    trusted = host == "huggingface.co" if origin else host in ALLOWED_DOWNLOAD_HOSTS
    if parsed.scheme != "https" or not trusted or parsed.username \
            or parsed.password or parsed.port not in (None, 443):
        raise ValueError(
            "Untrusted HTTPS Hugging Face download host: " +
            str(host or "<no host>")
        )
    return host


class _SafeHfRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_hf_download_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def open_hf_request(request: Request):
    _validate_hf_download_url(request.full_url, origin=True)
    return build_opener(_SafeHfRedirect()).open(request, timeout=45)


@dataclass(frozen=True)
class DownloadReport:
    status: str
    purpose: str
    path: str | None
    sha256: str
    revision: str | None
    expected_size_bytes: int | None
    bytes_present: int
    license_acknowledged_for_this_download: bool
    transfer_authorized: bool
    download_started: bool
    model_verified: bool
    model_loaded: bool
    gpu_offload_executed: bool
    inference_executed: bool
    cpu_fallback_selected: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    choice: ModelChoice, status: str, *, path: Path | None = None,
    bytes_present: int = 0, authorized: bool = False, license_ok: bool = False,
    started: bool = False, verified: bool = False, error: str | None = None,
) -> DownloadReport:
    return DownloadReport(
        status=status, purpose=choice.purpose,
        path=str(path) if path else None, sha256=choice.expected_sha256,
        revision=choice.revision,
        expected_size_bytes=choice.expected_size_bytes,
        bytes_present=bytes_present,
        license_acknowledged_for_this_download=license_ok,
        transfer_authorized=authorized, download_started=started,
        model_verified=verified, model_loaded=False,
        gpu_offload_executed=False, inference_executed=False,
        cpu_fallback_selected=False, error=error,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(BUFFER), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_path(root: Path, choice: ModelChoice) -> Path:
    # Every segment was validated by the model catalog. Reject fabricated
    # choices as well, including an absolute/reparented file path.
    segments = choice.path.split("/")
    if len(segments) != 5 or any(
        not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}", s)
        for s in segments
    ) or tuple(segments) != (
        choice.family, choice.model, choice.version,
        choice.format, choice.quantization,
    ):
        raise ValueError("Unsafe hierarchical catalog choice")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]+\.gguf", choice.filename):
        raise ValueError("Unsafe GGUF filename")
    return root / MODEL_ROOT / choice.path / choice.filename


def _artifact_policy(choice: ModelChoice) -> bool:
    return bool(
        choice.source == "huggingface"
        and REPO_ID.fullmatch(choice.repo_id)
        and choice.format == "gguf"
        and choice.license_acknowledgement_required
        and isinstance(choice.revision, str) and GIT_SHA.fullmatch(choice.revision)
        and SHA256.fullmatch(choice.expected_sha256)
        and type(choice.expected_size_bytes) is int
        and choice.expected_size_bytes >= 1024 * 1024
    )


def _manifest(choice: ModelChoice) -> dict:
    return {
        "schema": "ruth-gguf-download-v1",
        "repo_id": choice.repo_id,
        "source": choice.source,
        "revision": choice.revision,
        "filename": choice.filename,
        "sha256": choice.expected_sha256,
        "size_bytes": choice.expected_size_bytes,
        "format": choice.format, "quantization": choice.quantization,
        "license": choice.license,
    }


def download_pinned_model(
    choice: ModelChoice, *, project_root: Path = PROJECT_ROOT,
    allow_download: bool = False, acknowledge_model_license: bool = False,
    fetch_metadata: Callable[[str], dict] | None = None,
    open_request: Callable[[Request], Any] = open_hf_request,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> DownloadReport:
    """Fail-closed: explicit network and license decisions required each call."""
    if not allow_download:
        return _result(choice, "model_download_not_authorized",
                       error="Use --allow-model-download for a network transfer")
    if not acknowledge_model_license:
        return _result(choice, "model_license_acknowledgement_required",
                       authorized=True,
                       error="Review model-owner terms and explicitly acknowledge them")
    if not _artifact_policy(choice):
        return _result(
            choice, "model_metadata_not_pinned", authorized=True,
            license_ok=True, error="Immutable revision, expected byte count and SHA256 required"
        )
    root = Path(project_root)
    try:
        final = _local_path(root, choice)
        model_root = root / MODEL_ROOT
        # Check all ancestors inside the path, not just final destination.
        current = model_root
        for section in (*choice.path.split("/"),):
            if current.is_symlink():
                raise ValueError("Symlink in model storage path")
            current = current / section
        if current.is_symlink() or final.is_symlink():
            raise ValueError("Symlink in model storage path")
        if final.exists():
            if final.is_file() and final.stat().st_size == choice.expected_size_bytes \
                    and _sha256_file(final) == choice.expected_sha256:
                return _result(
                    choice, "model_already_verified", path=final,
                    bytes_present=choice.expected_size_bytes,
                    authorized=True, license_ok=True, verified=True,
                )
            raise ValueError("Existing model path has different or invalid content")
    except (OSError, ValueError) as exc:
        return _result(choice, "unsafe_or_existing_model",
                       authorized=True, license_ok=True,
                       error=str(exc)[:240])

    # Reverify immutable commit, expected HF LFS checksum and size BEFORE
    # opening the large /resolve/<commit>/<filename> endpoint.
    remotely = resolve_model_metadata(
        choice, authorize_metadata_network=True, fetch=fetch_metadata)
    if remotely["status"] != "source_metadata_verified_license_pending":
        return _result(
            choice, "model_source_not_verified", authorized=True,
            license_ok=True, error=remotely.get("error", remotely["status"]))
    if remotely["revision"] != choice.revision or \
            remotely["sha256"] != choice.expected_sha256 or \
            remotely["size_bytes"] != choice.expected_size_bytes:
        return _result(choice, "model_source_not_verified",
                       authorized=True, license_ok=True,
                       error="Remote metadata does not match immutable model pin")
    # Authentication/gated access and source-level license acceptance
    # are separate features. A public quantization does not waive terms.
    if remotely.get("gated") not in (False, None):
        return _result(choice, "gated_model_not_supported", authorized=True,
                       license_ok=True, error="Gated HF models need separate authentication")

    part = final.with_name(final.name + ".part")
    sidecar = final.with_name(final.name + ".part.json")
    completed = final.with_name(final.name + ".manifest.json")
    lock = final.with_name(final.name + ".download.lock")
    policy = _manifest(choice)
    try:
        if model_root.is_symlink():
            raise ValueError("Model directory symlink")
        final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        for p in (part, sidecar, completed, lock):
            if p.is_symlink():
                raise ValueError("Model staging symlink")
        if completed.exists() or lock.exists():
            raise ValueError("Existing manifest or active download lock")
        lock.mkdir(mode=0o700)
    except (OSError, ValueError) as exc:
        return _result(choice, "model_download_path_blocked",
                       authorized=True, license_ok=True, error=str(exc)[:240])

    started = False
    try:
        if part.exists() != sidecar.exists():
            raise ValueError("Partial model and resume manifest must coexist")
        if sidecar.exists():
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if data != policy:
                raise ValueError("Partial download belongs to different revision/hash")
        else:
            # Sidecar is written first, then data. A crash between them
            # leaves an empty part that can be safely resumed/restarted.
            sidecar.write_text(json.dumps(policy, sort_keys=True) + "\n",
                               encoding="utf-8")
            part.touch(exist_ok=False)
        offset = part.stat().st_size
        expected = choice.expected_size_bytes
        if offset > expected:
            raise ValueError("Partial model exceeds pinned byte count")
        if disk_usage(root).free < (expected - offset) + SPARE_BYTES:
            return _result(
                choice, "model_insufficient_disk", path=part,
                bytes_present=offset, authorized=True, license_ok=True,
                error="Require remaining download bytes plus 128 MiB headroom")
        if offset < expected:
            url = ("https://huggingface.co/" + choice.repo_id +
                   "/resolve/" + choice.revision + "/" +
                   quote(choice.filename, safe="") + "?download=true")
            request = Request(url, headers={
                "User-Agent": "Ruth-pinned-GGUF-download/1",
                "Accept-Encoding": "identity",
                **({"Range": f"bytes={offset}-"} if offset else {}),
            })
            started = True
            with open_request(request) as response:
                # Defense in depth: validate the final URL even for injected
                # transports, not only redirects from the real urllib opener.
                _validate_hf_download_url(response.geturl())
                status = response.status
                if status != (206 if offset else 200):
                    raise ValueError("Unexpected HTTP status; resumptions require 206")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ValueError("Compressed HTTP content not allowed")
                if offset:
                    marker = response.headers.get("Content-Range", "")
                    match = RANGE_RE.fullmatch(marker)
                    if not match or (
                        int(match.group(1)) != offset or
                        int(match.group(2)) != expected - 1 or
                        int(match.group(3)) != expected
                    ):
                        raise ValueError("Invalid HTTP partial-content range")
                length = response.headers.get("Content-Length")
                if length is not None and (
                    not length.isdecimal() or
                    int(length) != expected - offset
                ):
                    raise ValueError("HTTP response size differs from pin")
                total = offset
                with part.open("ab") as stream:
                    while data := response.read(BUFFER):
                        total += len(data)
                        if total > expected:
                            raise ValueError("Remote transfer exceeded pinned file size")
                        stream.write(data)
        current = part.stat().st_size
        if current != expected:
            raise ValueError("Incomplete transfer; resumable partial retained")
        with part.open("rb") as probe:
            if probe.read(4) != b"GGUF":
                raise ValueError("File lacks GGUF magic bytes")
        digest = _sha256_file(part)
        if digest != choice.expected_sha256:
            return _result(
                choice, "model_checksum_mismatch", path=part,
                bytes_present=current, authorized=True, license_ok=True,
                started=started, error="Full-file SHA256 differs from catalog pin"
            )
        # Manifest first, then rename. The full pinned file is never exposed
        # under the final model name unless its digest matches.
        manifest_tmp = completed.with_name(completed.name + ".tmp")
        if manifest_tmp.exists() or manifest_tmp.is_symlink():
            raise ValueError("Unexpected existing manifest staging path")
        manifest_tmp.write_text(json.dumps({
            **policy, "verified": True,
            "license_acknowledged_for_this_download": True,
            "model_loaded": False, "inference_executed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if final.exists() or final.is_symlink():
            raise ValueError("Model appeared during transfer; not overwriting")
        part.rename(final)
        manifest_tmp.rename(completed)
        sidecar.unlink(missing_ok=True)
        return _result(
            choice, "model_download_verified", path=final,
            bytes_present=expected, authorized=True, license_ok=True,
            started=started, verified=True,
        )
    except (OSError, ValueError, TypeError, KeyError, HTTPError, URLError,
            json.JSONDecodeError) as exc:
        return _result(
            choice, "model_download_interrupted_or_blocked", path=part,
            bytes_present=part.stat().st_size if part.is_file() else 0,
            authorized=True, license_ok=True, started=started,
            error=type(exc).__name__ + ": " + str(exc)[:220],
        )
    finally:
        lock.rmdir()
