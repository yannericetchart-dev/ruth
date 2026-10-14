"""CUDA source build planning and opt-in automatic acquisition entry point.

--plan is read-only. --prepare automates the pinned Git download with one
explicit authorization, never installs Python packages or compiles a model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable

from .dependencies import PROJECT_ROOT
from .install_plan import (
    InstallationPlan, SOURCE_COMMIT, SOURCE_REPOSITORY,
    LLAMA_CPP_SUBMODULE_COMMIT, NATIVE_BUILD_CMAKE_ARGS,
    SOURCE_BUILD_REQUIRES, build_installation_plan,
)

MIN_CMAKE = (3, 21, 0)
SOURCE_RELATIVE_PATH = Path(".ruth-src") / "llama-cpp-python"
SOURCE_GIT_REMOTE = SOURCE_REPOSITORY + ".git"
_CMAKE_PATTERN = re.compile(r"^cmake version (\d+)\.(\d+)(?:\.(\d+))?", re.M)


@dataclass(frozen=True)
class BuildPrerequisite:
    name: str
    status: str  # ok, missing, incompatible, probe_failed
    detected: str | None
    remediation: str | None


@dataclass(frozen=True)
class SourceBuildPlan:
    strategy: str
    runtime_preflight: str
    toolkit_status: str
    tools: tuple[BuildPrerequisite, ...]
    source_directory: str
    source_status: str
    source_commit_expected: str
    source_commit_actual: str | None
    llama_cpp_commit_expected: str
    llama_cpp_commit_actual: str | None
    checkout_commands: tuple[str, ...]
    cuda_cmake_args: str
    python_build_requirements: tuple[str, ...]
    dependency_lock_ready: bool
    source_verified: bool
    build_prerequisites_met: bool
    ready_to_build: bool
    download_authorized: bool
    build_authorized: bool
    fallback_automatically_selected: bool
    observations: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_cmake_version(value: str) -> tuple[int, int, int] | None:
    match = _CMAKE_PATTERN.search(value)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _check_cli_tool(
    name: str,
    *,
    executable: str,
    args: tuple[str, ...],
    run: Callable[..., object],
    minimum: tuple[int, int, int] | None = None,
) -> BuildPrerequisite:
    try:
        result = run([executable, *args], text=True, capture_output=True,
                     check=False, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return BuildPrerequisite(
            name, "probe_failed", None,
            "Non è stato possibile eseguire il controllo: " + type(exc).__name__)
    if result.returncode:
        return BuildPrerequisite(
            name, "probe_failed", None,
            "Comando terminato con codice " + str(result.returncode))
    output = result.stdout.strip()
    if minimum:
        version = parse_cmake_version(output)
        if version is None:
            return BuildPrerequisite(name, "probe_failed", output[:140],
                                     "Formato versione CMake non riconosciuto.")
        if version < minimum:
            return BuildPrerequisite(name, "incompatible", ".".join(map(str, version)),
                                     "Richiesto CMake >=3.21; Ubuntu: controlla "
                                     "'apt-cache policy cmake' e aggiorna solo dopo approvazione.")
        return BuildPrerequisite(name, "ok", ".".join(map(str, version)), None)
    return BuildPrerequisite(name, "ok", output.splitlines()[0][:140] if output else None, None)


def _git_read(
    run: Callable[..., object], directory: Path, args: list[str]
) -> tuple[str | None, str | None]:
    try:
        # Avoid index-refresh writes during git status: diagnostic is read-only.
        env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
        result = run(["git", "-C", str(directory), *args],
                     capture_output=True, text=True, check=False, timeout=15,
                     env=env)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, type(exc).__name__
    if result.returncode:
        return None, f"exit={result.returncode}: {(result.stderr or '')[:180].strip()}"
    return result.stdout.strip(), None


def inspect_checkout(
    source: Path,
    *,
    git_available: bool,
    run: Callable[..., object] = subprocess.run,
) -> tuple[str, str | None, str | None]:
    """Verify exact nested commits, remote and clean tree. Never modify checkout."""
    if source.is_symlink():
        return "unsafe_symlink", None, None
    if not source.exists():
        return "missing", None, None
    if not source.is_dir() or not git_available:
        return "invalid", None, None
    llama_dir = source / "vendor" / "llama.cpp"
    if llama_dir.is_symlink() or not llama_dir.is_dir():
        return "missing_submodule", None, None
    head, head_err = _git_read(run, source, ["rev-parse", "HEAD"])
    if head_err or not head:
        return "probe_failed", None, None
    nested, nested_err = _git_read(run, llama_dir, ["rev-parse", "HEAD"])
    if nested_err or not nested:
        return "missing_submodule", head, None
    if head != SOURCE_COMMIT or nested != LLAMA_CPP_SUBMODULE_COMMIT:
        return "commit_mismatch", head, nested
    remote, remote_err = _git_read(run, source, ["remote", "get-url", "origin"])
    if remote_err or remote not in (SOURCE_REPOSITORY, SOURCE_GIT_REMOTE):
        return "remote_mismatch", head, nested
    # The parent status reports tracked submodule modifications too.
    changes, changes_err = _git_read(run, source, ["status", "--porcelain", "--untracked-files=all"])
    if changes_err or changes:
        return "dirty", head, nested
    if not (source / "pyproject.toml").is_file() or not (llama_dir / "CMakeLists.txt").is_file():
        return "incomplete", head, nested
    return "verified", head, nested


def create_source_build_plan(
    *,
    install_plan: InstallationPlan | None = None,
    project_root: Path = PROJECT_ROOT,
    source_dir: Path | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., object] = subprocess.run,
) -> SourceBuildPlan:
    """Check prerequisites and source identity, not Python host pip."""
    if install_plan is None:
        install_plan = build_installation_plan()
    source = source_dir if source_dir is not None else project_root / SOURCE_RELATIVE_PATH

    tools: list[BuildPrerequisite] = []
    git = which("git")
    if git is None:
        tools.append(BuildPrerequisite("git", "missing", None,
            "Ubuntu/Debian: sudo apt install git (solo previa approvazione)."))
    else:
        tools.append(_check_cli_tool(
            "git", executable=git, args=("--version",), run=run))
    cmake = which("cmake")
    if cmake is None:
        tools.append(BuildPrerequisite("CMake", "missing", None,
            "Ubuntu/Debian: sudo apt install cmake (>=3.21, previa approvazione)."))
    else:
        tools.append(_check_cli_tool(
            "CMake", executable=cmake, args=("--version",), run=run,
            minimum=MIN_CMAKE))
    cxx = which("c++")
    if cxx is None:
        tools.append(BuildPrerequisite("C++ compiler", "missing", None,
            "Ubuntu/Debian: sudo apt install build-essential (previa approvazione)."))
    else:
        tools.append(_check_cli_tool(
            "C++ compiler", executable=cxx, args=("--version",), run=run))

    # Fix the generator to Unix Makefiles to prevent scikit-build-core from
    # silently fetching Ninja. Ruth never installs host build tools implicitly.
    make = which("make")
    if make is None:
        tools.append(BuildPrerequisite("GNU Make", "missing", None,
            "Ubuntu/Debian: sudo apt install make (previa approvazione)."))
    else:
        tools.append(_check_cli_tool(
            "GNU Make", executable=make, args=("--version",), run=run))

    git_ready = tools[0].status == "ok"
    source_status, source_head, llama_head = inspect_checkout(
        source, git_available=git_ready, run=run)

    commands = (
        "mkdir -p .ruth-src",
        "git clone --no-checkout " + SOURCE_GIT_REMOTE
        + " .ruth-src/llama-cpp-python",
        "git -C .ruth-src/llama-cpp-python checkout --detach " + SOURCE_COMMIT,
        "git -C .ruth-src/llama-cpp-python submodule update --init --recursive",
        "git -C .ruth-src/llama-cpp-python rev-parse HEAD",
        "git -C .ruth-src/llama-cpp-python/vendor/llama.cpp rev-parse HEAD",
    )
    tool_ok = all(item.status == "ok" for item in tools)
    prereqs = bool(install_plan.primary_prerequisites_met and tool_ok)
    notes = (
        "DEFAULT --plan is read-only; --prepare automatically acquires and verifies "
        "pinned sources with one explicit download authorization.",
        "Review Git source provenance and pinned parent/submodule revisions before builds.",
        "Pin with SHA256 every Python runtime/build wheel and its complete transitive closure.",
        "Use ONLY .venv/bin/python for Python dependencies, never host pip/site-packages.",
        "Source build later: CMAKE_GENERATOR='Unix Makefiles' "
        "CMAKE_BUILD_PARALLEL_LEVEL=2 CMAKE_ARGS='" + NATIVE_BUILD_CMAKE_ARGS +
        " -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc' "
        "CUDA_HOME=/usr/local/cuda-13.2; use --no-build-isolation and approved locks.",
        "No build, pip, network request or fallback performed by this plan.",
        "Wheel cu132 is an explicit MANUAL fallback; CPU-only is forbidden.",
        "No LlamaDiskCache in first runtime; adapter enforcement must be tested.",
    )
    return SourceBuildPlan(
        strategy="cuda-toolkit-native-build",
        runtime_preflight=install_plan.runtime_preflight,
        toolkit_status=str(install_plan.toolkit["status"]),
        tools=tuple(tools),
        source_directory=str(source),
        source_status=source_status,
        source_commit_expected=SOURCE_COMMIT,
        source_commit_actual=source_head,
        llama_cpp_commit_expected=LLAMA_CPP_SUBMODULE_COMMIT,
        llama_cpp_commit_actual=llama_head,
        checkout_commands=commands,
        cuda_cmake_args=NATIVE_BUILD_CMAKE_ARGS,
        python_build_requirements=SOURCE_BUILD_REQUIRES,
        dependency_lock_ready=False,
        source_verified=(source_status == "verified"),
        build_prerequisites_met=prereqs,
        ready_to_build=False,  # locked dependencies & verified source still required
        download_authorized=False,
        build_authorized=False,
        fallback_automatically_selected=False,
        observations=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Ruth pinned CUDA source preflight and acquisition")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true",
                         help="Read-only preflight (default); no network or disk writes")
    actions.add_argument("--prepare", action="store_true",
                         help="Authorize and automate pinned Git source acquisition (network/disk)")
    args = parser.parse_args()
    report = create_source_build_plan()
    if not args.prepare:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    from .source_acquisition import acquire_pinned_sources
    result = acquire_pinned_sources(report)
    # Verify the resulting checkout automatically; no second user command.
    checked = create_source_build_plan() if result.source_verified else None
    accepted = bool(checked and checked.source_verified and result.source_verified)
    print(json.dumps({
        "operation": "prepare_sources",
        "acquisition": result.to_dict(),
        "plan_after": checked.to_dict() if checked else None,
        "build_and_pip_not_started": True,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
