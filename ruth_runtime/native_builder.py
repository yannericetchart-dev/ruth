"""Offline CUDA native wheel builder for Ruth, with immutable input pins.

The explicit --build-native action:
 * DOES NOT install into Ruth's .venv (pip install uses --target in staging).
 * DOES NOT download packages or models, alter NVIDIA drivers, or fall back to CPU.
 * Copies verified source into staging because upstream CMake can write to source.
 * Produces a versioned, SHA256-recorded CUDA wheel, pending actual GPU probe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Any
from zipfile import ZipFile, BadZipFile

from .dependencies import PROJECT_ROOT
from .dependency_resolution import (
    CANDIDATE_NAME, WHEELHOUSE, inspect_wheel, normalize_project_name,
)
from .install_plan import (
    SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT, WHEEL_VERSION,
)
from .source_build_plan import SourceBuildPlan, create_source_build_plan
from .venv_packages import EnvironmentReport, inspect_python_environment

BUILD_ROOT = Path(".ruth-build")
BUILD_OUTPUT = f"native-cuda-{WHEEL_VERSION}-sm89"
PIN_MANIFEST = Path("locks") / "python-cuda-3.12.build-inputs.json"
BUILD_TOOLS = frozenset({"scikit-build-core", "packaging", "pathspec"})
MINIMUM_FREE_BYTES = 8 * 1024**3
NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
CUDA_HOME = Path("/usr/local/cuda-13.2")
CUDA_ARGUMENTS = (
    "-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 "
    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DLLAVA_BUILD=OFF"
)


@dataclass(frozen=True)
class NativeWheelResult:
    status: str
    source_verified: bool
    isolated_venv_verified: bool
    pinned_wheels_verified: bool
    native_build_started: bool
    wheel_file: str | None
    wheel_sha256: str | None
    ggml_cuda_library_packaged: bool
    gpu_offload_executed: bool
    inference_executed: bool
    installed_in_ruth_venv: bool
    fallback_automatically_selected: bool
    risk_todo_open: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    status: str, *, source: bool, venv: bool, pins: bool = False,
    started: bool = False, wheel: str | None = None,
    digest: str | None = None, cuda_lib: bool = False,
    error: str | None = None,
) -> NativeWheelResult:
    return NativeWheelResult(
        status=status, source_verified=source,
        isolated_venv_verified=venv, pinned_wheels_verified=pins,
        native_build_started=started, wheel_file=wheel, wheel_sha256=digest,
        ggml_cuda_library_packaged=cuda_lib, gpu_offload_executed=False,
        inference_executed=False, installed_in_ruth_venv=False,
        fallback_automatically_selected=False, risk_todo_open=True,
        error=error,
    )


def verify_pinned_build_inputs(root: Path) -> tuple[bool, tuple[Path, ...], str | None]:
    """Check the committed pin reference, local candidate manifest and bytes.

    This does not bless the vulnerable runtime dependency or install it.
    """
    candidate = root / WHEELHOUSE / CANDIDATE_NAME
    candidate_manifest = candidate / "candidate-lock.json"
    candidate_wheels = candidate / "wheels"
    pin_file = root / PIN_MANIFEST
    if any(p.is_symlink() for p in (
        root / WHEELHOUSE, candidate, candidate_manifest,
        candidate_wheels, pin_file
    )):
        return False, (), "Symlink in pin or candidate path"
    try:
        pins = json.loads(pin_file.read_text(encoding="utf-8"))
        manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
        if pins.get("schema") != "ruth-cuda-build-inputs-v1" \
                or pins.get("source_commit") != SOURCE_COMMIT \
                or pins.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT \
                or pins.get("llama_cpp_python_version") != WHEEL_VERSION:
            raise ValueError("Pinned build manifest does not match Ruth source policy")
        if manifest.get("schema") != "ruth-python-candidate-v1" \
                or manifest.get("approved") is not False \
                or manifest.get("dependency_lock_ready") is not False \
                or manifest.get("source_commit") != SOURCE_COMMIT \
                or manifest.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT \
                or manifest.get("python") != "3.12" \
                or manifest.get("cuda_architecture") != "89":
            raise ValueError("Candidate manifest does not match pinned sources")
        pinned = pins["files"]
        staged = manifest["files"]
        if not isinstance(pinned, list) or len(pinned) != 8 \
                or not isinstance(staged, list) or len(staged) != 8:
            raise ValueError("Exactly eight pinned wheel records are required")
        def rows(values):
            return {normalize_project_name(r["name"]): r for r in values}
        by_name = rows(pinned)
        if len(by_name) != len(pinned) or rows(staged) != by_name:
            raise ValueError("Candidate differs from reviewed pinned wheel hashes")
        if {normalize_project_name(n) for n in by_name} != {
            "diskcache", "jinja2", "markupsafe", "numpy", "packaging",
            "pathspec", "scikit-build-core", "typing-extensions"
        }:
            raise ValueError("Unexpected pinned package set")
        paths: list[Path] = []
        for row in pinned:
            name = row["filename"]
            if not isinstance(name, str) or name != Path(name).name:
                raise ValueError("Unsafe pinned wheel filename")
            wheel = candidate_wheels / name
            if wheel.is_symlink() or asdict(inspect_wheel(wheel)) != row:
                raise ValueError("Local wheel changed from pinned bytes: " + name)
            if normalize_project_name(row["name"]) in BUILD_TOOLS:
                paths.append(wheel)
        if len(paths) != len(BUILD_TOOLS):
            raise ValueError("Missing build tool wheel")
        if sorted(p.name for p in candidate_wheels.iterdir()) != \
                sorted(r["filename"] for r in pinned):
            raise ValueError("Unrecognized extra wheel artifact")
        return True, tuple(sorted(paths)), None
    except (OSError, ValueError, TypeError, KeyError, AttributeError,
            BadZipFile, UnicodeError, json.JSONDecodeError) as exc:
        return False, (), type(exc).__name__ + ": " + str(exc)[:200]


def inspect_cuda_wheel(path: Path) -> tuple[bool, str | None, str | None]:
    """Require packaged GGML CUDA backend. This is NOT GPU execution proof."""
    try:
        wheel = inspect_wheel(path)
        if normalize_project_name(wheel.name) != "llama-cpp-python" \
                or wheel.version != WHEEL_VERSION:
            return False, None, "Built wheel name/version mismatch"
        with ZipFile(path) as archive:
            members = archive.namelist()
            cuda = any(
                name.startswith("llama_cpp/lib/") and
                Path(name).name.startswith("libggml-cuda.so")
                for name in members
            )
            core = any(
                name.startswith("llama_cpp/lib/") and
                Path(name).name.startswith("libllama.so")
                for name in members
            )
        if not cuda or not core:
            return False, None, "Built wheel is missing libggml-cuda.so or libllama.so"
        return True, wheel.sha256, None
    except (OSError, BadZipFile, ValueError) as exc:
        return False, None, type(exc).__name__


def _safe_source_copy(source: Path, dest: Path) -> bool:
    if not source.is_dir() or source.is_symlink():
        return False
    # A staging build must never dereference symlinks out of the pinned checkout.
    # Git's .git entries and local bytecode are not part of the build input.
    def ignore(directory: str, names: list[str]) -> set[str]:
        return {
            name for name in names
            if name == ".git" or name == "__pycache__" or name.endswith(".pyc")
        }
    shutil.copytree(source, dest, symlinks=True, ignore=ignore)
    return not any(p.is_symlink() for p in dest.rglob("*"))


def build_native_cuda_wheel(
    *, project_root: Path = PROJECT_ROOT,
    environment: EnvironmentReport | None = None,
    source_plan: SourceBuildPlan | None = None,
    run: Callable[..., Any] = subprocess.run,
    source_copy: Callable[[Path, Path], bool] = _safe_source_copy,
    pins_check: Callable[[Path], tuple[bool, tuple[Path, ...], str | None]] = verify_pinned_build_inputs,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> NativeWheelResult:
    """Explicit one-shot native build, offline; install into Ruth .venv is separate."""
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    plan = source_plan if source_plan is not None else create_source_build_plan()
    venv_ok = (
        env.interpreter_in_project_venv
        and env.python_version.startswith("3.12.")
        and Path(env.project_root).resolve() == root.resolve()
        and Path(env.interpreter).resolve() ==
            (root / ".venv/bin/python").resolve()
    )
    source = Path(plan.source_directory)
    source_ok = (
        bool(plan.source_verified and plan.build_prerequisites_met)
        and plan.source_status == "verified"
        and plan.source_commit_actual == SOURCE_COMMIT
        and plan.llama_cpp_commit_actual == LLAMA_CPP_SUBMODULE_COMMIT
        and source.resolve() == (root / ".ruth-src/llama-cpp-python").resolve()
        and not source.is_symlink()
    )
    if not venv_ok or not source_ok:
        return _result("blocked_preflight", source=source_ok, venv=venv_ok,
                       error="Verified project venv and exact pinned CUDA checkout required")
    if not NVCC.is_file():
        return _result("missing_nvcc", source=True, venv=True,
                       error="CUDA compiler 13.2 is unavailable")
    try:
        if disk_usage(root).free < MINIMUM_FREE_BYTES:
            return _result("insufficient_disk", source=True, venv=True,
                           error="At least 8 GiB free required for staged build")
    except OSError:
        return _result("disk_probe_failed", source=True, venv=True)
    pins_ok, tool_wheels, pin_error = pins_check(root)
    if not pins_ok:
        return _result("wheel_pins_mismatch", source=True, venv=True,
                       error=pin_error)

    buildroot = root / BUILD_ROOT
    output = buildroot / BUILD_OUTPUT
    if buildroot.is_symlink() or output.exists() or output.is_symlink():
        return _result("output_already_exists", source=True, venv=True, pins=True,
                       error="Never overwrite an existing native wheel build")
    try:
        buildroot.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock = buildroot / ".native-cuda-build.lock"
        lock.mkdir(mode=0o700)
    except (FileExistsError, OSError):
        return _result("build_busy_or_unwritable", source=True, venv=True, pins=True)
    stage: Path | None = None
    started = False
    try:
        if output.exists() or output.is_symlink():
            return _result("output_already_exists", source=True, venv=True, pins=True)
        stage = Path(tempfile.mkdtemp(prefix=".native-cuda-stage-", dir=buildroot))
        tools_dir, stage_wheels, copied = (
            stage / "build-tools", stage / "wheels", stage / "source"
        )
        stage_wheels.mkdir(mode=0o700)
        if not source_copy(source, copied):
            return _result("unsafe_source_copy", source=True, venv=True, pins=True,
                           error="A symlink or invalid source was detected")
        python = str(root / ".venv/bin/python")
        base = {
            **os.environ, "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_NO_INDEX": "1",
        }
        tool_cmd = [
            python, "-m", "pip", "--isolated", "install", "--no-index",
            "--no-deps", "--no-cache-dir", "--no-compile",
            "--target", str(tools_dir), *(str(p) for p in tool_wheels),
        ]
        # Stage build frontend only. No diskcache or llama-cpp-python install.
        try:
            dep_result = run(tool_cmd, cwd=str(root), env=base, text=True,
                             capture_output=True, check=False, timeout=180)
        except (OSError, subprocess.SubprocessError) as exc:
            return _result("build_tools_failed", source=True, venv=True, pins=True,
                           error=type(exc).__name__)
        if dep_result.returncode:
            return _result("build_tools_failed", source=True, venv=True, pins=True,
                           error="Offline build tooling stage exit " +
                                 str(dep_result.returncode))

        build_env = {
            **base,
            "PYTHONPATH": str(tools_dir),
            "CUDA_HOME": str(CUDA_HOME),
            "CUDACXX": str(NVCC),
            "CMAKE_CUDA_COMPILER": str(NVCC),
            "CMAKE_GENERATOR": "Unix Makefiles",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
            "CMAKE_ARGS": CUDA_ARGUMENTS,
        }
        wheel_cmd = [
            python, "-m", "pip", "--isolated", "wheel", "--no-index",
            "--no-deps", "--no-build-isolation", "--no-cache-dir",
            "--wheel-dir", str(stage_wheels), str(copied),
        ]
        started = True
        try:
            wheel_result = run(wheel_cmd, cwd=str(root), env=build_env,
                               text=True, capture_output=True, check=False,
                               timeout=7200)
        except (OSError, subprocess.SubprocessError) as exc:
            return _result("cuda_compile_failed", source=True, venv=True,
                           pins=True, started=started,
                           error=type(exc).__name__)
        if wheel_result.returncode:
            details = ((wheel_result.stdout or "") + "\n" +
                       (wheel_result.stderr or "")).strip()[-1600:]
            return _result("cuda_compile_failed", source=True, venv=True,
                           pins=True, started=True,
                           error="CUDA wheel build exited " +
                                 str(wheel_result.returncode) +
                                 "; compiler diagnostics (tail): " + details)
        items = list(stage_wheels.iterdir())
        if len(items) != 1 or not items[0].is_file() or items[0].is_symlink() \
                or items[0].suffix != ".whl":
            return _result("unexpected_build_outputs", source=True, venv=True,
                           pins=True, started=True)
        valid, digest, wheel_error = inspect_cuda_wheel(items[0])
        if not valid:
            return _result("cuda_backend_missing", source=True, venv=True,
                           pins=True, started=True, error=wheel_error)
        (stage / "build-result.json").write_text(
            json.dumps({
                "schema": "ruth-native-cuda-wheel-v1",
                "source_commit": SOURCE_COMMIT,
                "llama_cpp_commit": LLAMA_CPP_SUBMODULE_COMMIT,
                "cuda_architecture": "89", "cuda_home": str(CUDA_HOME),
                "cmake_args": CUDA_ARGUMENTS,
                "wheel_file": items[0].name, "wheel_sha256": digest,
                "ggml_cuda_library_packaged": True,
                "gpu_offload_executed": False,
                "inference_executed": False,
                "installed_in_ruth_venv": False,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Keep only wheel+manifest, never a second copy of native sources,
        # build tools or intermediate compiler objects in the final artifact.
        if tools_dir.exists():
            shutil.rmtree(tools_dir)
        if copied.exists():
            shutil.rmtree(copied)
        stage.rename(output)
        stage = None
        return _result("native_cuda_wheel_built", source=True, venv=True,
                       pins=True, started=True,
                       wheel=str(output / "wheels" / items[0].name),
                       digest=digest, cuda_lib=True)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        lock.rmdir()
