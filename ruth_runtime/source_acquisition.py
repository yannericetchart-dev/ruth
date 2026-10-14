"""Automatic, fail-closed acquisition of Ruth's exactly pinned CUDA sources.

Called only by an explicit source-prepare action or by a future authorized
orchestrator. Never installs packages, compiles code or chooses a CPU/wheel
fallback. Writes only to Ruth's ignored .ruth-src/ area.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

from .install_plan import SOURCE_COMMIT, SOURCE_REPOSITORY, LLAMA_CPP_SUBMODULE_COMMIT
from .source_build_plan import (
    SOURCE_RELATIVE_PATH, SourceBuildPlan, create_source_build_plan, inspect_checkout,
)

UPSTREAM_TAG = "v0.3.36"
SUBMODULE_PATH = "vendor/llama.cpp"
SOURCE_REMOTE = SOURCE_REPOSITORY + ".git"
SUBMODULE_REMOTE = "https://github.com/ggml-org/llama.cpp.git"


@dataclass(frozen=True)
class SourceAcquisitionResult:
    status: str  # already_verified, acquired_verified, blocked, failed
    source_directory: str
    source_status_before: str
    source_status_after: str
    source_verified: bool
    source_commit: str | None
    llama_cpp_commit: str | None
    download_performed: bool
    changed_existing_checkout: bool
    fallback_automatically_selected: bool
    build_started: bool
    python_install_started: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _report(
    status: str,
    source: Path,
    before: str,
    after: str,
    verified: bool = False,
    parent_commit: str | None = None,
    child_commit: str | None = None,
    downloaded: bool = False,
    error: str | None = None,
) -> SourceAcquisitionResult:
    return SourceAcquisitionResult(
        status=status,
        source_directory=str(source),
        source_status_before=before,
        source_status_after=after,
        source_verified=verified,
        source_commit=parent_commit,
        llama_cpp_commit=child_commit,
        download_performed=downloaded,
        changed_existing_checkout=False,
        fallback_automatically_selected=False,
        build_started=False,
        python_install_started=False,
        error=error,
    )


def _git_env() -> dict[str, str]:
    """Disable prompts, inherited Git URL rewrites and interactive credentials."""
    return dict(
        os.environ,
        GIT_TERMINAL_PROMPT="0",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_OPTIONAL_LOCKS="0",
        GIT_LFS_SKIP_SMUDGE="1",
    )


def _git_cmd(*parts: str) -> list[str]:
    return [
        "git",
        "-c", "protocol.file.allow=never",
        "-c", "protocol.ext.allow=never",
        "-c", "core.hooksPath=/dev/null",
        *parts,
    ]


def acquire_pinned_sources(
    plan: SourceBuildPlan | None = None,
    *,
    project_root: Path | None = None,
    run: Callable[..., object] = subprocess.run,
) -> SourceAcquisitionResult:
    """Perform the pinned Git steps automatically, only on an explicit call.

    A clean existing checkout is reused; anything unexpected is left untouched.
    A new checkout is staged then promoted only after commit verification.
    """
    if plan is None:
        plan = create_source_build_plan()
    source = Path(plan.source_directory)
    before = plan.source_status
    expected_source = (
        project_root / SOURCE_RELATIVE_PATH
        if project_root is not None else source
    )
    if source != expected_source or source.name != "llama-cpp-python" \
            or source.parent.name != ".ruth-src" or not source.is_absolute():
        return _report("blocked", source, before, before,
                       error="Source path is not the approved Ruth .ruth-src checkout")
    if before == "verified":
        return _report(
            "already_verified", source, before, before, True,
            plan.source_commit_actual, plan.llama_cpp_commit_actual)
    if before != "missing":
        return _report(
            "blocked", source, before, before,
            error="An existing unverified checkout will not be modified")
    if not plan.build_prerequisites_met or plan.runtime_preflight != "ok" \
            or plan.toolkit_status != "ok" \
            or not any(t.name == "git" and t.status == "ok" for t in plan.tools):
        return _report(
            "blocked", source, before, before,
            error="Native build prerequisites and Git must be verified first")

    # Reject symlinks to outside the project and pre-existing source paths.
    parent = source.parent
    if parent.is_symlink() or source.is_symlink() or source.exists():
        return _report(
            "blocked", source, before, "existing",
            error="Refusing symlink or existing checkout path")
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        return _report("failed", source, before, before,
                       error="Cannot create source area: " + type(exc).__name__)
    if parent.is_symlink() or not parent.is_dir():
        return _report("blocked", source, before, before,
                       error="Source area must be a real directory")

    lock_dir = parent / ".ruth-source-acquire.lock"
    try:
        lock_dir.mkdir(mode=0o700)
    except FileExistsError:
        return _report(
            "blocked", source, before, before,
            error="Another source acquisition or a stale lock is present")
    except OSError as exc:
        return _report("failed", source, before, before,
                       error="Cannot acquire lock: " + type(exc).__name__)

    stage: Path | None = None
    downloaded = False
    try:
        # Recheck under the lock. Never replace an unexpected target.
        if source.is_symlink() or source.exists():
            return _report("blocked", source, before, "existing",
                           error="Target appeared while acquiring lock")
        stage = Path(tempfile.mkdtemp(prefix=".ruth-source-stage-", dir=parent))
        env = _git_env()
        steps: tuple[tuple[list[str], int], ...] = (
            (_git_cmd(
                "clone", "--no-checkout", "--single-branch", "--depth", "1",
                "--branch", UPSTREAM_TAG, SOURCE_REMOTE, str(stage)
            ), 600),
            (_git_cmd(
                "-C", str(stage), "checkout", "--detach", SOURCE_COMMIT
            ), 120),
            (_git_cmd(
                "-C", str(stage), "submodule", "update", "--init", "--depth", "1",
                "--", SUBMODULE_PATH
            ), 600),
        )
        for index, (command, timeout) in enumerate(steps):
            print(
                "Ruth: " + ("acquisizione repository" if index == 0 else
                            "checkout commit fissato" if index == 1 else
                            "acquisizione submodule llama.cpp"),
                file=sys.stderr, flush=True,
            )
            # An attempt to clone is a download even if it exits with an error.
            if index == 0:
                downloaded = True
            try:
                result = run(command, env=env, text=True,
                             capture_output=True, check=False, timeout=timeout)
            except (OSError, subprocess.SubprocessError) as exc:
                return _report(
                    "failed", source, before, "missing", downloaded=downloaded,
                    error=f"Git stage {index + 1}: {type(exc).__name__}")
            if result.returncode:
                # Output can contain sensitive user/URL data: don't echo it.
                return _report(
                    "failed", source, before, "missing", downloaded=downloaded,
                    error=f"Git stage {index + 1} exited {result.returncode}")

        # This checks the commit identity, submodule, remote, dirty tree and files.
        checkout_state, head, child = inspect_checkout(
            stage, git_available=True, run=run)
        if checkout_state != "verified":
            return _report(
                "failed", source, before, checkout_state,
                parent_commit=head, child_commit=child, downloaded=downloaded,
                error="Checkout verification failed: " + checkout_state)
        if source.is_symlink() or source.exists():
            return _report(
                "blocked", source, before, "existing", downloaded=downloaded,
                error="Checkout target appeared during Git acquisition")
        try:
            stage.rename(source)
            stage = None
        except OSError as exc:
            return _report(
                "failed", source, before, "missing", downloaded=downloaded,
                error="Cannot promote checkout: " + type(exc).__name__)

        # Verify the promoted path one more time.
        final_status, final_parent, final_child = inspect_checkout(
            source, git_available=True, run=run)
        if final_status != "verified":
            return _report(
                "failed", source, before, final_status,
                parent_commit=final_parent, child_commit=final_child,
                downloaded=downloaded,
                error="Promoted checkout failed verification; inspect manually")
        return _report(
            "acquired_verified", source, before, "verified", True,
            final_parent, final_child, downloaded=downloaded)
    finally:
        if stage is not None and stage.exists():
            # Only our unique staging directory is removed; never the target.
            shutil.rmtree(stage)
        lock_dir.rmdir()
