"""Phase 02: offline, read-only review of one pinned NVIDIA CUDA wheel.

This module NEVER downloads or installs software. The upstream wheel metadata
has been checked against the official GitHub release v0.3.36-cu132.
Python dependencies and hashes must be locked before any installation action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import re
import shutil
import sys
from typing import Callable

from .dependencies import DependencyReport, inspect_dependencies

WHEEL_VERSION = "0.3.36"
CUDA_WHEEL = "cu132"
WHEEL_FILENAME = "llama_cpp_python-0.3.36-py3-none-manylinux_2_35_x86_64.whl"
WHEEL_SOURCE = (
    "https://github.com/abetlen/llama-cpp-python/releases/download/"
    "v0.3.36-cu132/" + WHEEL_FILENAME
)
WHEEL_SHA256 = "808bc0344d77351efcd2aa19fc978799de9f07ce7313e135bbc202bf4aac136e"
WHEEL_SIZE_BYTES = 783614981
MIN_GLIBC = (2, 35)
# Names are upstream runtime requirements, NOT a lockfile or approved versions.
# Direct constraints from pyproject.toml at upstream tag v0.3.36.
# MarkupSafe is transitive via Jinja2. Versions and wheel SHA256s remain TO DO.
DIRECT_DEPENDENCIES = (
    "typing-extensions>=4.5.0",
    "numpy>=1.20.0",
    "diskcache>=5.6.1",
    "jinja2>=2.11.3",
)
TRANSITIVE_DEPENDENCIES = ("MarkupSafe>=2.0",)
UNLOCKED_DEPENDENCIES = (
    "numpy", "typing-extensions", "diskcache", "jinja2", "markupsafe"
)
DISKCACHE_ADVISORY = "GHSA-w8v5-vhqr-4h9v / CVE-2025-69872"
DISKCACHE_ADVISORY_URL = "https://github.com/advisories/GHSA-w8v5-vhqr-4h9v"



@dataclass(frozen=True)
class InstallationPlan:
    package: str
    version: str
    cuda_variant: str
    wheel_filename: str
    wheel_url: str
    sha256: str
    download_bytes: int
    interpreter: str
    runtime_preflight: str
    glibc: str
    glibc_compatible: bool
    filesystem_free_bytes: int | None
    wheel_download_fits: bool | None
    unpinned_dependencies: tuple[str, ...]
    direct_requirements: tuple[str, ...]
    transitive_requirements: tuple[str, ...]
    security_advisories: tuple[str, ...]
    disk_cache_policy: str
    dependency_lock_ready: bool
    installation_ready: bool
    eligible_for_review: bool
    installation_authorized: bool
    installation_implemented: bool
    notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_glibc_version(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", value.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def build_installation_plan(
    *,
    dependencies: DependencyReport | None = None,
    libc: tuple[str, str] | None = None,
    free_bytes: int | None = None,
    disk_usage: Callable[[str | Path], object] = shutil.disk_usage,
    project_dir: Path | None = None,
    interpreter: str | None = None,
) -> InstallationPlan:
    """Design review only: no network, file writes, GPU inference or pip.

    No attempt is made to choose an unapproved alternative or CPU wheel.
    A missing llama-cpp-python package is EXPECTED and not a preflight blocker.
    """
    if dependencies is None:
        dependencies = inspect_dependencies()
    if libc is None:
        libc = platform.libc_ver()
    parsed = parse_glibc_version(libc[1])
    libc_ok = libc[0].lower() in ("glibc", "gnu libc") and parsed is not None and parsed >= MIN_GLIBC

    if free_bytes is None:
        try:
            root = project_dir or Path(__file__).resolve().parents[1]
            free_bytes = int(disk_usage(root).free)
        except OSError:
            free_bytes = None
    fit = (free_bytes >= WHEEL_SIZE_BYTES) if free_bytes is not None else None

    # llama_cpp_python is not yet installed. That is exactly what this plan
    # prepares for; only platform/venv/driver/CUDA are gating prerequisites.
    preflight_names = {
        "GNU/Linux x86_64",
        "Python >=3.10 (Ruth .venv)",
        "Ruth .venv",
        "nvidia-smi",
        "GPU/driver NVIDIA",
        "CUDA supportata dal driver",
    }
    preflight = {
        item.name: item.status for item in dependencies.checks
        if item.name in preflight_names
    }
    prereqs_ok = all(preflight.get(name) == "ok" for name in preflight_names)

    notes = (
        "No downloads or installations are implemented by this command.",
        "The package is pinned to an official cu132 Linux wheel; no source/CPU fallback.",
        "The GPU compute capability and native CUDA execution remain unverified.",
        "Upstream runtime dependencies are NOT yet version/hash locked.",
        "diskcache <=5.6.3 has a known advisory (CVE-2025-69872): default pickle "
        "deserialization is unsafe if untrusted users can write cache files. "
        "The first Ruth runtime must not instantiate LlamaDiskCache.",
        "No user-writable disk cache or untrusted cache deserialization is allowed. "
        "Adapter-level enforcement/tests are not implemented yet.",
        "Downloading the wheel is not enough disk space for installation/extraction.",
        "Any download and package installation require a separate approved policy.",
    )
    return InstallationPlan(
        package="llama-cpp-python", version=WHEEL_VERSION,
        cuda_variant=CUDA_WHEEL, wheel_filename=WHEEL_FILENAME,
        wheel_url=WHEEL_SOURCE, sha256=WHEEL_SHA256,
        download_bytes=WHEEL_SIZE_BYTES,
        interpreter=interpreter or sys.executable,
        runtime_preflight=("ok" if prereqs_ok else "blocked"),
        glibc=libc[0] + " " + libc[1],
        glibc_compatible=bool(libc_ok),
        filesystem_free_bytes=free_bytes,
        wheel_download_fits=fit,
        unpinned_dependencies=UNLOCKED_DEPENDENCIES,
        direct_requirements=DIRECT_DEPENDENCIES,
        transitive_requirements=TRANSITIVE_DEPENDENCIES,
        security_advisories=(DISKCACHE_ADVISORY + " (" + DISKCACHE_ADVISORY_URL + ")",),
        disk_cache_policy=(
            "DISABLED for initial Ruth runtime: never create LlamaDiskCache; "
            "no loading deserialized cache files; requires adapter-level tests"
        ),
        dependency_lock_ready=False,
        installation_ready=False,
        eligible_for_review=bool(prereqs_ok and libc_ok and fit),
        installation_authorized=False,
        installation_implemented=False,
        notes=notes,
    )


def main() -> int:
    print(json.dumps(build_installation_plan().to_dict(), indent=2,
                     ensure_ascii=False, sort_keys=True))
    # A blocked or incomplete plan is not an installation error, but report
    # incompatibility through the exit code without modifying the environment.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
