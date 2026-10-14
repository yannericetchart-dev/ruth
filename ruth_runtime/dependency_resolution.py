"""Stage a candidate Python wheel lock without installing packages.

Ruth source CUDA build remains PRIMARY. The upstream llama-cpp-python wheel
is never downloaded by this resolver. PyPI wheels in an isolated staging
directory are fully inventoried and hashed, but require manual lock/security
approval before any installation. No hidden pip install or CPU fallback.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable
from zipfile import BadZipFile, ZipFile

from .dependencies import PROJECT_ROOT
from .install_plan import (
    DIRECT_DEPENDENCIES, SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT,
    SOURCE_BUILD_REQUIRES, DISKCACHE_ADVISORY,
)
from .venv_packages import EnvironmentReport, inspect_python_environment

WHEELHOUSE = Path(".ruth-wheelhouse")
CANDIDATE_NAME = "python-dependencies-candidate"
PYPI_INDEX = "https://pypi.org/simple/"
# Do not download llama-cpp-python: the approved primary route BUILDS it locally.
ROOT_REQUIREMENTS = DIRECT_DEPENDENCIES + SOURCE_BUILD_REQUIRES
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_WHEEL = re.compile(r"^[A-Za-z0-9_.+-]+\.whl$")


@dataclass(frozen=True)
class WheelFile:
    filename: str
    name: str
    version: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CandidateResolution:
    status: str
    venv_verified: bool
    approved_source_pins: bool
    downloaded: bool
    wheelhouse: str | None
    candidate_manifest: str | None
    wheels: tuple[WheelFile, ...]
    lock_approved: bool
    dependency_lock_ready: bool
    installation_started: bool
    compilation_started: bool
    fallback_automatically_selected: bool
    advisory: str
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    status: str, venv_ok: bool, source_ok: bool,
    *, downloaded: bool = False, output: Path | None = None,
    wheels: tuple[WheelFile, ...] = (), error: str | None = None,
) -> CandidateResolution:
    return CandidateResolution(
        status=status, venv_verified=venv_ok, approved_source_pins=source_ok,
        downloaded=downloaded, wheelhouse=str(output) if output else None,
        candidate_manifest=str(output / "candidate-lock.json") if output else None,
        wheels=wheels, lock_approved=False, dependency_lock_ready=False,
        installation_started=False, compilation_started=False,
        fallback_automatically_selected=False,
        advisory=DISKCACHE_ADVISORY + ": all upstream diskcache releases "
        "are affected; do not use LlamaDiskCache or untrusted disk cache data.",
        error=error,
    )


def normalize_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def inspect_wheel(path: Path) -> WheelFile:
    """Read wheel metadata without executing or importing package code."""
    if not SAFE_WHEEL.fullmatch(path.name) or path.is_symlink():
        raise ValueError("Unexpected or symlink wheel filename")
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    with ZipFile(path) as zf:
        metadata_files = [entry for entry in zf.namelist()
                          if entry.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise ValueError("Wheel lacks unique dist-info/METADATA")
        with zf.open(metadata_files[0]) as src:
            data = src.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("Excessive wheel metadata")
        headers = Parser().parsestr(data.decode("utf-8"))
        name, version = headers.get("Name"), headers.get("Version")
        if not name or not version or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version):
            raise ValueError("Malformed distribution metadata")
        dist_parts = path.name.removesuffix(".whl").split("-")
        if len(dist_parts) < 5 or normalize_project_name(dist_parts[0]) != normalize_project_name(name) or dist_parts[1] != version:
            raise ValueError("Wheel filename and METADATA disagree")
    return WheelFile(path.name, name, version, checksum.hexdigest(), path.stat().st_size)


def stage_dependency_candidate(
    *,
    authorize_network: bool = False,
    environment: EnvironmentReport | None = None,
    source_verified: bool,
    project_root: Path = PROJECT_ROOT,
    run: Callable[..., object] = subprocess.run,
) -> CandidateResolution:
    """Resolve complete dependency graph into a quarantined candidate wheelhouse.

    This does NOT claim that pip's graph or package safety has been reviewed.
    The caller must explicitly authorize this Python dependency download.
    """
    env = environment if environment is not None else inspect_python_environment()
    venv_ok = env.interpreter_in_project_venv
    if not venv_ok or not source_verified:
        return _result("blocked_preflight", venv_ok, source_verified,
                       error="A verified isolated venv and pinned source checkout are required")
    if not authorize_network:
        return _result("download_not_authorized", venv_ok, source_verified,
                       error="Use explicit dependency download authorization")
    if not env.python_version.startswith("3.12."):
        return _result("blocked_python", venv_ok, source_verified,
                       error="Candidate target is CPython 3.12 only")
    root = Path(project_root)
    target_parent = root / WHEELHOUSE
    final = target_parent / CANDIDATE_NAME
    if target_parent.is_symlink() or final.is_symlink() or final.exists():
        return _result("blocked_existing_candidate", venv_ok, source_verified,
                       error="Existing candidate or symlink will never be overwritten")
    # The environment report is associated with THIS Ruth tree and interpreter.
    if Path(env.project_root).resolve() != root.resolve() \
            or Path(env.interpreter).resolve() != (root / ".venv/bin/python").resolve():
        return _result("blocked_interpreter", venv_ok, source_verified,
                       error="Interpreter is not from the requested Ruth project")
    if target_parent.is_symlink():
        return _result("blocked_symlink", venv_ok, source_verified)
    try:
        target_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        return _result("filesystem_failed", venv_ok, source_verified,
                       error=type(exc).__name__)
    lock = target_parent / ".candidate-download.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError:
        return _result("busy", venv_ok, source_verified,
                       error="Candidate resolver already running or stale lock")
    except OSError as exc:
        return _result("filesystem_failed", venv_ok, source_verified,
                       error=type(exc).__name__)
    staging: Path | None = None
    attempted = False
    try:
        if final.exists() or final.is_symlink():
            return _result("blocked_existing_candidate", venv_ok, source_verified)
        staging = Path(tempfile.mkdtemp(dir=target_parent, prefix=".candidate-stage-"))
        wheels_dir = staging / "wheels"
        wheels_dir.mkdir(mode=0o700)
        # --isolated ignores user pip settings, --only-binary prevents sdist
        # builds, --no-deps is intentionally ABSENT: resolve transitives.
        command = [
            str(root / ".venv/bin/python"), "-m", "pip", "--isolated",
            "download", "--only-binary=:all:", "--disable-pip-version-check",
            "--no-input", "--no-cache-dir", "--index-url", PYPI_INDEX,
            "--dest", str(wheels_dir), *ROOT_REQUIREMENTS,
        ]
        attempted = True
        try:
            result = run(command, capture_output=True, text=True, check=False,
                         timeout=600, cwd=str(root),
                         env={**os.environ, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        except (OSError, subprocess.SubprocessError) as exc:
            return _result("resolver_failed", venv_ok, source_verified,
                           downloaded=attempted, error=type(exc).__name__)
        if result.returncode:
            return _result("resolver_failed", venv_ok, source_verified,
                           downloaded=attempted,
                           error=f"pip download failed (exit {result.returncode}); no installation")
        files = list(wheels_dir.iterdir())
        if not files or any(not p.is_file() or p.is_symlink() or not p.name.endswith(".whl")
                            for p in files):
            return _result("invalid_artifacts", venv_ok, source_verified,
                           downloaded=attempted,
                           error="Only complete regular wheel artifacts are allowed")
        try:
            wheels = tuple(sorted((inspect_wheel(p) for p in files),
                                  key=lambda w: normalize_project_name(w.name)))
        except (OSError, ValueError, UnicodeError, BadZipFile) as exc:
            return _result("invalid_artifacts", venv_ok, source_verified,
                           downloaded=attempted, error=type(exc).__name__)
        normalized = [normalize_project_name(w.name) for w in wheels]
        if len(set(normalized)) != len(normalized) or "llama-cpp-python" in normalized:
            return _result("invalid_artifacts", venv_ok, source_verified,
                           downloaded=attempted,
                           error="Duplicate package or unexpected prebuilt llama-cpp-python wheel")
        requested = ("numpy", "typing-extensions", "diskcache",
                     "jinja2", "scikit-build-core")
        if not all(x in normalized for x in requested):
            return _result("invalid_artifacts", venv_ok, source_verified,
                           downloaded=attempted,
                           error="One or more required root packages are absent")
        manifest = {
            "schema": "ruth-python-candidate-v1",
            "approved": False,
            "dependency_lock_ready": False,
            "python": "3.12",
            "index": PYPI_INDEX,
            "source_commit": SOURCE_COMMIT,
            "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
            "cuda_architecture": "89",
            "requested": ROOT_REQUIREMENTS,
            "files": [asdict(x) for x in wheels],
            "security_review_required": [DISKCACHE_ADVISORY],
            "notes": "Candidate only. Hashes prove downloaded bytes; independent "
                     "upstream provenance, dependency closure and security "
                     "review are required before install.",
        }
        (staging / "candidate-lock.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        if final.exists() or final.is_symlink():
            return _result("blocked_existing_candidate", venv_ok, source_verified,
                           downloaded=attempted)
        staging.rename(final)
        staging = None
        return _result("candidate_review_required", venv_ok, source_verified,
                       downloaded=attempted, output=final, wheels=wheels)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        lock.rmdir()
