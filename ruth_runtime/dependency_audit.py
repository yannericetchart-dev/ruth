"""Audit a staged Python wheel candidate without installing or executing wheel code.

Checks local wheel bytes and names against the candidate JSON, PyPI release
digests and yank state (metadata-only network), and pip's offline dry-run
resolver for CPython 3.12 Linux x86_64. Security review remains a separate
gate: diskcache 5.6.3 is affected by GHSA-w8v5-vhqr-4h9v.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Callable, Any
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from zipfile import BadZipFile

from .dependencies import PROJECT_ROOT
from .dependency_resolution import (
    CANDIDATE_NAME, PYPI_INDEX, ROOT_REQUIREMENTS, WHEELHOUSE,
    inspect_wheel, normalize_project_name,
)
from .install_plan import (
    SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT, DISKCACHE_ADVISORY,
)
from .venv_packages import EnvironmentReport, inspect_python_environment

_REQUIRED = frozenset({
    "numpy", "typing-extensions", "jinja2", "diskcache", "scikit-build-core",
})
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+!-]*$")
_HEX = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AuditReport:
    status: str
    candidate: str
    files_verified: int
    wheel_hashes_verified: bool
    pypi_metadata_verified: bool
    dependency_closure_verified: bool
    wheel_compatibility_verified: bool
    security_review_passed: bool
    lock_approved: bool
    installation_authorized: bool
    installation_started: bool
    build_started: bool
    fallback_automatically_selected: bool
    issues: tuple[str, ...]
    warning: str

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_pypi_metadata(name: str, version: str) -> dict:
    """Only read fixed PyPI JSON metadata; no wheel download or third-party hosts."""
    if not _SAFE_NAME.fullmatch(name) or not _SAFE_VERSION.fullmatch(version):
        raise ValueError("Invalid PyPI name/version")
    url = (f"https://pypi.org/pypi/{quote(name)}/{quote(version)}/json")
    request = Request(url, headers={"User-Agent": "Ruth-candidate-audit/1"})
    with urlopen(request, timeout=20) as response:
        if urlsplit(response.url).hostname != "pypi.org":
            raise ValueError("PyPI redirected outside approved metadata endpoint")
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("PyPI release JSON exceeds size limit")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("urls"), list):
        raise ValueError("Malformed PyPI release metadata")
    return data


def _audit(
    *, status: str, candidate: Path, count: int = 0,
    hashed: bool = False, pypi: bool = False, closure: bool = False,
    tags: bool = False, issues: list[str] | tuple[str, ...] = (),
) -> AuditReport:
    return AuditReport(
        status=status, candidate=str(candidate),
        files_verified=count, wheel_hashes_verified=hashed,
        pypi_metadata_verified=pypi, dependency_closure_verified=closure,
        wheel_compatibility_verified=tags,
        security_review_passed=False, lock_approved=False,
        installation_authorized=False, installation_started=False,
        build_started=False, fallback_automatically_selected=False,
        issues=tuple(issues),
        warning=DISKCACHE_ADVISORY + ": diskcache <=5.6.3 has no upstream "
                "patched release. This audit never approves installation. "
                "LlamaDiskCache must remain disabled and be tested in the adapter.",
    )


def audit_candidate(
    *,
    project_root: Path = PROJECT_ROOT,
    environment: EnvironmentReport | None = None,
    pypi_metadata: Callable[[str, str], dict] = fetch_pypi_metadata,
    run: Callable[..., Any] = subprocess.run,
    network_authorized: bool = False,
) -> AuditReport:
    """All gates fail closed. A digest match never grants permission to install."""
    root = Path(project_root)
    candidate = root / WHEELHOUSE / CANDIDATE_NAME
    env = environment if environment is not None else inspect_python_environment()
    if not env.interpreter_in_project_venv or \
            Path(env.project_root).resolve() != root.resolve() or \
            Path(env.interpreter).resolve() != (root / ".venv/bin/python").resolve():
        return _audit(status="blocked_venv", candidate=candidate,
                      issues=["Not the isolated Ruth Python interpreter"])
    if not network_authorized:
        return _audit(status="metadata_network_not_authorized", candidate=candidate,
                      issues=["Explicit authorization for PyPI metadata lookups required"])
    wheels_dir = candidate / "wheels"
    manifest_file = candidate / "candidate-lock.json"
    if candidate.is_symlink() or wheels_dir.is_symlink() or \
            not candidate.is_dir() or not wheels_dir.is_dir() or \
            manifest_file.is_symlink() or not manifest_file.is_file():
        return _audit(status="missing_candidate", candidate=candidate,
                      issues=["Candidate directory or manifest absent/unsafe"])
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return _audit(status="invalid_manifest", candidate=candidate,
                      issues=["Cannot parse candidate-lock.json"])
    if not isinstance(manifest, dict) or \
            manifest.get("schema") != "ruth-python-candidate-v1" or \
            manifest.get("approved") is not False or \
            manifest.get("dependency_lock_ready") is not False or \
            manifest.get("source_commit") != SOURCE_COMMIT or \
            manifest.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT or \
            manifest.get("cuda_architecture") != "89" or \
            manifest.get("python") != "3.12" or \
            manifest.get("index") != PYPI_INDEX or \
            tuple(manifest.get("requested", ())) != ROOT_REQUIREMENTS:
        return _audit(status="invalid_manifest", candidate=candidate,
                      issues=["Candidate policy, pinned source or requested roots mismatch"])
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows or len(rows) > 100:
        return _audit(status="invalid_manifest", candidate=candidate,
                      issues=["Missing or excessive wheel records"])
    try:
        paths = list(wheels_dir.iterdir())
        filenames = {p.name for p in paths}
        expected = {r["filename"] for r in rows}
        if len(filenames) != len(rows) or filenames != expected or \
                len(expected) != len(rows) or \
                any(not p.is_file() or p.is_symlink() for p in paths):
            raise ValueError("Unexpected, missing, symlink or duplicate wheel files")
        verified = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Malformed wheel record")
            name = row["filename"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Invalid filename")
            w = inspect_wheel(wheels_dir / name)
            if asdict(w) != row or not _HEX.fullmatch(w.sha256):
                raise ValueError("Wheel checksum, size, version or name mismatch: " + name)
            verified.append(w)
        by_name = {normalize_project_name(w.name): w for w in verified}
        if len(by_name) != len(verified) or not _REQUIRED.issubset(by_name) \
                or "llama-cpp-python" in by_name:
            raise ValueError("Required wheels missing, duplicates, or CPU fallback wheel")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, BadZipFile) as exc:
        return _audit(status="invalid_wheels", candidate=candidate,
                      issues=[str(exc)[:220]])
    # Independent matching of PyPI *release metadata*: hashes, length and yank state.
    for w in verified:
        try:
            info = pypi_metadata(w.name, w.version)
            listed = [i for i in info["urls"] if
                      i.get("filename") == w.filename and
                      i.get("packagetype") == "bdist_wheel"]
            if len(listed) != 1 or listed[0].get("yanked") or \
                    listed[0].get("digests", {}).get("sha256") != w.sha256 or \
                    listed[0].get("size") != w.size_bytes:
                raise ValueError("PyPI digest, size, wheel type or yank state mismatch")
        except (OSError, ValueError, KeyError, TypeError, AttributeError,
                json.JSONDecodeError) as exc:
            return _audit(
                status="pypi_mismatch", candidate=candidate,
                count=len(verified), hashed=True,
                issues=[w.filename + ": " + type(exc).__name__ + " " + str(exc)[:140]])
        except Exception as exc:
            return _audit(
                status="pypi_probe_failed", candidate=candidate,
                count=len(verified), hashed=True,
                issues=[w.filename + ": " + type(exc).__name__])
    # Offline pip dry-run is authoritative for PEP 508 markers, extras,
    # Python 3.12/glibc wheel tags, root requirements and transitive closure.
    # Explicit exact-version pins force ALL staged wheels into the resolution.
    requirements = list(ROOT_REQUIREMENTS)
    requirements.extend(f"{w.name}=={w.version}" for w in verified)
    try:
        with tempfile.TemporaryDirectory(prefix="ruth-pip-audit-") as tmp:
            report_path = Path(tmp) / "pip-report.json"
            command = [
                str(root / ".venv/bin/python"), "-m", "pip", "--isolated",
                "install", "--dry-run", "--ignore-installed", "--no-index",
                "--no-cache-dir", "--disable-pip-version-check", "--no-input",
                "--only-binary=:all:", "--find-links", str(wheels_dir),
                "--report", str(report_path), *requirements,
            ]
            result = run(command, capture_output=True, text=True, check=False,
                         timeout=180, cwd=str(root))
            if result.returncode != 0 or not report_path.is_file():
                return _audit(
                    status="dependency_closure_failed", candidate=candidate,
                    count=len(verified), hashed=True, pypi=True,
                    issues=["pip offline resolver rejected staged candidate (exit " +
                            str(result.returncode) + ")"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            installs = report.get("install")
            if not isinstance(installs, list):
                raise ValueError("Invalid pip report")
            selected = {}
            for entry in installs:
                meta = entry.get("metadata", {})
                name = normalize_project_name(meta.get("name", ""))
                version = meta.get("version")
                if name in selected or not version:
                    raise ValueError("Duplicate or incomplete pip report")
                selected[name] = version
            if selected != {n: w.version for n, w in by_name.items()}:
                raise ValueError("Resolved package set differs from the staged wheels")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError,
            subprocess.SubprocessError) as exc:
        return _audit(status="dependency_closure_failed", candidate=candidate,
                      count=len(verified), hashed=True, pypi=True,
                      issues=[type(exc).__name__ + ": " + str(exc)[:150]])
    return _audit(
        status="security_review_required", candidate=candidate,
        count=len(verified), hashed=True, pypi=True, closure=True, tags=True,
        issues=["CVE-2025-69872: diskcache cache deserialization policy not "
                "yet enforced by a tested Ruth CUDA adapter"],
    )
