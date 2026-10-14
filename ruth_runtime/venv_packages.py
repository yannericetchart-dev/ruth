"""Ruth Python environment inventory. Read-only and Python-stdlib only.

Never consult host site-packages when outside the project venv.
Installed package metadata is NOT proof of CUDA, hashes or LLM inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
import re
import sys
from typing import Callable

from .dependencies import PROJECT_ROOT
from .install_plan import DIRECT_DEPENDENCIES, TRANSITIVE_DEPENDENCIES, WHEEL_VERSION

LOCK_PATH = Path("locks") / "python-cuda-3.12.lock.json"
# Specs are upstream constraints, NOT a reviewed fully hashed dependency lock.
REQUIREMENTS = (
    ("llama-cpp-python", f"=={WHEEL_VERSION}", "runtime"),
    ("numpy", ">=1.20.0", "runtime"),
    ("typing-extensions", ">=4.5.0", "runtime"),
    ("diskcache", ">=5.6.1", "runtime"),
    ("jinja2", ">=2.11.3", "runtime"),
    ("MarkupSafe", ">=2.0", "runtime"),
    ("scikit-build-core", ">=0.9.2", "build"),
)


@dataclass(frozen=True)
class PackageState:
    name: str
    requirement: str
    role: str
    installed_version: str | None
    status: str  # missing / incompatible / present_unverified / inspection_failed / blocked


@dataclass(frozen=True)
class EnvironmentReport:
    project_root: str
    interpreter: str
    interpreter_in_project_venv: bool
    python_version: str
    venv_status: str
    lock_path: str
    lock_status: str
    packages: tuple[PackageState, ...]
    missing_runtime: tuple[str, ...]
    missing_build: tuple[str, ...]
    runtime_dependencies_present: bool
    runtime_dependencies_verified: bool
    python_packages_installable: bool
    backend_gpu_verified: bool
    ruth_runtime_ready: bool
    remediation: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_python_environment(
    *,
    project_root: Path = PROJECT_ROOT,
    prefix: Path | None = None,
    base_prefix: Path | None = None,
    interpreter: str | None = None,
    python_version: tuple[int, ...] | None = None,
    package_version: Callable[[str], str] = metadata.version,
) -> EnvironmentReport:
    root = Path(project_root)
    venv = root / ".venv"
    actual_prefix = Path(prefix) if prefix is not None else Path(sys.prefix)
    host_prefix = Path(base_prefix) if base_prefix is not None else Path(sys.base_prefix)
    interp = interpreter if interpreter is not None else sys.executable
    py = python_version if python_version is not None else sys.version_info[:3]
    cfg = venv / "pyvenv.cfg"
    try:
        contents = cfg.read_text(encoding="utf-8") if cfg.is_file() else ""
    except OSError:
        contents = ""
    global_packages = any(
        re.fullmatch(r"include-system-site-packages\s*=\s*true", line.strip(),
                     flags=re.IGNORECASE)
        for line in contents.splitlines()
    )
    correct = (
        not venv.is_symlink()
        and cfg.is_file()
        and actual_prefix.resolve() == venv.resolve()
        and actual_prefix.resolve() != host_prefix.resolve()
        and not global_packages
        and tuple(py[:2]) >= (3, 10)
    )
    status = "ok" if correct else "invalid_or_inactive"
    states: list[PackageState] = []
    for name, requirement, role in REQUIREMENTS:
        if not correct:
            states.append(PackageState(name, requirement, role, None, "blocked"))
            continue
        try:
            installed = package_version(name)
        except metadata.PackageNotFoundError:
            installed = None
            condition = "missing"
        except Exception:
            installed = None
            condition = "inspection_failed"
        else:
            condition = (
                "incompatible" if name == "llama-cpp-python"
                and installed != WHEEL_VERSION else "present_unverified"
            )
        states.append(PackageState(name, requirement, role, installed, condition))
    missing_runtime = tuple(
        p.name for p in states
        if p.role == "runtime" and p.status != "present_unverified"
    )
    missing_build = tuple(
        p.name for p in states
        if p.role == "build" and p.status != "present_unverified"
    )
    lock = root / LOCK_PATH
    # A user-created file with the expected name is not an audited lock.
    lock_status = "unreviewed" if lock.is_file() else "missing"
    guidance = (
        "Usa il launcher di Ruth dalla directory del repository e la .venv "
        "locale; se assente: ./ruth setup. Nessuna dipendenza Python globale."
        if not correct else
        "Manca un lockfile completamente verificato con versioni e SHA256 "
        "dei wheel runtime e build. Nessun pip automatico finché il gate "
        "di sicurezza non è chiuso."
    )
    return EnvironmentReport(
        project_root=str(root),
        interpreter=interp,
        interpreter_in_project_venv=correct,
        python_version=".".join(str(i) for i in py),
        venv_status=status,
        lock_path=str(lock),
        lock_status=lock_status,
        packages=tuple(states),
        missing_runtime=missing_runtime,
        missing_build=missing_build,
        runtime_dependencies_present=correct and not missing_runtime,
        runtime_dependencies_verified=False,
        python_packages_installable=False,
        backend_gpu_verified=False,
        ruth_runtime_ready=False,
        remediation=guidance,
    )
