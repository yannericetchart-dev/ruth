"""Fail-closed offline installation of Ruth's already-built CUDA wheel.

Uses an isolated --target install and import/GPU-feature smoke test FIRST.
An explicitly authorized operation then promotes to a contained directory in
Ruth's .venv and atomically activates it using a .pth file. No pip install
touches global or normal site-packages, no download, rebuild, CPU fallback.

The diskcache CVE remains an open TODO. Installation requires explicit,
separate acknowledgement of that residual risk; mere mitigations are not
approval. GPU support detection is NOT a model inference test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .dependency_resolution import CANDIDATE_NAME, WHEELHOUSE
from .install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT, WHEEL_VERSION
from .native_builder import (
    BUILD_ROOT, BUILD_OUTPUT, inspect_cuda_wheel, verify_pinned_build_inputs,
)
from .venv_packages import EnvironmentReport, inspect_python_environment

NATIVE_LOCK = Path("locks/native-cuda-0.3.36-sm89-wheel.json")
RUNTIME_FOLDER = "ruth-native-cuda-0.3.36-sm89"
PTH_FILE = "zz_ruth_native_cuda.pth"


@dataclass(frozen=True)
class NativeInstallResult:
    status: str
    venv_verified: bool
    candidate_wheels_verified: bool
    native_wheel_verified: bool
    residual_diskcache_risk_acknowledged: bool
    installation_started: bool
    installed_in_ruth_venv: bool
    gpu_offload_supported: bool
    gpu_offload_executed: bool
    inference_executed: bool
    wheel_sha256: str | None
    native_runtime_directory: str | None
    rollback_performed: bool
    fallback_automatically_selected: bool
    diskcache_todo_open: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _result(
    status: str, *, venv: bool = False, pins: bool = False,
    wheel: bool = False, risk: bool = False, started: bool = False,
    installed: bool = False, supported: bool = False,
    checksum: str | None = None, location: Path | None = None,
    rolled_back: bool = False, error: str | None = None,
) -> NativeInstallResult:
    return NativeInstallResult(
        status=status, venv_verified=venv,
        candidate_wheels_verified=pins, native_wheel_verified=wheel,
        residual_diskcache_risk_acknowledged=risk,
        installation_started=started, installed_in_ruth_venv=installed,
        gpu_offload_supported=supported, gpu_offload_executed=False,
        inference_executed=False, wheel_sha256=checksum,
        native_runtime_directory=str(location) if location else None,
        rollback_performed=rolled_back, fallback_automatically_selected=False,
        diskcache_todo_open=True, error=error,
    )


def inspect_native_install_inputs(
    *, project_root: Path = PROJECT_ROOT,
    environment: EnvironmentReport | None = None,
    pins_check: Callable[[Path], tuple[bool, tuple[Path, ...], str | None]] = verify_pinned_build_inputs,
    inspect_built: Callable[[Path], tuple[bool, str | None, str | None]] = inspect_cuda_wheel,
) -> tuple[NativeInstallResult, tuple[Path, ...]]:
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    correct = bool(
        env.interpreter_in_project_venv
        and env.python_version.startswith("3.12.")
        and Path(env.project_root).resolve() == root.resolve()
        and Path(env.interpreter).resolve() == (root / ".venv/bin/python").resolve()
        and not (root / ".venv").is_symlink()
    )
    if not correct:
        return _result("blocked_venv", error="Ruth's isolated Python 3.12 venv required"), ()
    # Fail closed when an existing distribution could precede the staged
    # location on sys.path and shadow the pinned runtime packages.
    runtime_names = {
        "llama-cpp-python", "numpy", "typing-extensions", "diskcache",
        "jinja2", "markupsafe",
    }
    preexisting = [
        p.name for p in env.packages
        if p.name.lower() in runtime_names and p.installed_version is not None
    ]
    if preexisting:
        return _result(
            "preexisting_runtime_packages", venv=True,
            error="Existing .venv distributions require separate reconciliation: " +
                  ", ".join(preexisting)), ()
    pins_ok, _tools, reason = pins_check(root)
    if not pins_ok:
        return _result("pinned_dependency_mismatch", venv=True, error=reason), ()
    wheelroot = root / BUILD_ROOT / BUILD_OUTPUT
    lock_path = root / NATIVE_LOCK
    result_manifest = wheelroot / "build-result.json"
    if any(p.is_symlink() for p in (
        root / BUILD_ROOT, wheelroot, wheelroot / "wheels",
        lock_path, result_manifest,
        root / WHEELHOUSE, root / WHEELHOUSE / CANDIDATE_NAME,
    )):
        return _result("unsafe_artifact_path", venv=True, pins=True), ()
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        built = json.loads(result_manifest.read_text(encoding="utf-8"))
        if lock.get("schema") != "ruth-native-wheel-lock-v1" or \
                lock.get("source_commit") != SOURCE_COMMIT or \
                lock.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT or \
                lock.get("python") != "3.12" or \
                lock.get("cuda_architecture") != "89":
            raise ValueError("Native wheel source/platform lock mismatch")
        filename = lock["wheel_filename"]
        if not isinstance(filename, str) or Path(filename).name != filename or \
                filename != "llama_cpp_python-0.3.36-py3-none-linux_x86_64.whl":
            raise ValueError("Native wheel filename mismatch")
        checksum = lock["wheel_sha256"]
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Native SHA256 pin is invalid")
        artifact = wheelroot / "wheels" / filename
        if artifact.is_symlink() or not artifact.is_file():
            raise ValueError("Pinned native wheel is missing or symlinked")
        ok, actual_hash, error = inspect_built(artifact)
        if not ok or actual_hash != checksum:
            raise ValueError("Native CUDA wheel checksum/backend mismatch: " + str(error))
        if built.get("wheel_file") != filename or \
                built.get("wheel_sha256") != checksum or \
                built.get("source_commit") != SOURCE_COMMIT or \
                built.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT or \
                built.get("cuda_architecture") != "89" or \
                built.get("ggml_cuda_library_packaged") is not True or \
                built.get("installed_in_ruth_venv") is not False:
            raise ValueError("Native build-result.json does not match pinned wheel")
        candidate = root / WHEELHOUSE / CANDIDATE_NAME / "wheels"
        manifest = json.loads((
            root / WHEELHOUSE / CANDIDATE_NAME / "candidate-lock.json"
        ).read_text(encoding="utf-8"))
        if manifest.get("approved") is not False:
            raise ValueError("Unexpectedly modified candidate manifest")
        deps = tuple(candidate / row["filename"] for row in manifest["files"])
        if len(deps) != 8 or any(p.is_symlink() or not p.is_file() for p in deps):
            raise ValueError("Candidate package files missing")
        return _result(
            "ready_for_explicit_install_authorization",
            venv=True, pins=True, wheel=True, checksum=checksum
        ), (*deps, artifact)
    except (OSError, ValueError, KeyError, TypeError, AttributeError,
            json.JSONDecodeError) as exc:
        return _result(
            "native_wheel_unverified", venv=True, pins=True,
            error=type(exc).__name__ + ": " + str(exc)[:220]
        ), ()


# Smoke must run in a FRESH isolated Ruth interpreter; the first use of
# -S prevents accidental imports from the original .venv site-packages.
PROBE_PROGRAM = (
    "import json,sys,pathlib;"
    "root=pathlib.Path(sys.argv[1]).resolve();"
    "sys.path.insert(0,str(root)) if sys.argv[2]=='stage' else None;"
    "import llama_cpp;"
    "origin=pathlib.Path(llama_cpp.__file__).resolve();"
    "assert origin.is_relative_to(root), 'unexpected llama_cpp installation';"
    "assert llama_cpp.__version__=='0.3.36', 'wrong llama_cpp version';"
    "gpu=llama_cpp.llama_supports_gpu_offload();"
    "print(json.dumps({'origin':str(origin),'cuda_supported':bool(gpu)}));"
    "sys.exit(0 if gpu else 4)"
)


def _probe(
    python: Path, package_dir: Path, *,
    staged: bool, run: Callable[..., Any],
    root: Path,
) -> bool:
    cmd = [
        str(python), "-I", *(["-S"] if staged else []),
        "-c", PROBE_PROGRAM, str(package_dir),
        "stage" if staged else "active",
    ]
    result = run(
        cmd, cwd=str(root), text=True, capture_output=True,
        check=False, timeout=90,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if result.returncode:
        return False
    try:
        lines = (result.stdout or "").splitlines()
        reply = json.loads(lines[-1]) if lines else {}
        return reply.get("cuda_supported") is True and \
               Path(reply.get("origin", "/")).resolve().is_relative_to(
                   package_dir.resolve())
    except (ValueError, OSError, TypeError):
        return False


def install_native_cuda(
    *, project_root: Path = PROJECT_ROOT,
    environment: EnvironmentReport | None = None,
    accept_diskcache_risk: bool = False,
    authorize_install: bool = False,
    run: Callable[..., Any] = subprocess.run,
    pins_check: Callable[[Path], tuple[bool, tuple[Path, ...], str | None]] = verify_pinned_build_inputs,
    inspect_built: Callable[[Path], tuple[bool, str | None, str | None]] = inspect_cuda_wheel,
    probe: Callable[..., bool] = _probe,
) -> NativeInstallResult:
    """Promote offline staged packages only after two explicit permissions."""
    root = Path(project_root)
    initial, wheels = inspect_native_install_inputs(
        project_root=root, environment=environment,
        pins_check=pins_check, inspect_built=inspect_built,
    )
    if initial.status != "ready_for_explicit_install_authorization":
        return initial
    if not authorize_install or not accept_diskcache_risk:
        return _result(
            "explicit_install_and_risk_consent_required",
            venv=True, pins=True, wheel=True, checksum=initial.wheel_sha256,
            error="Both install authorization and diskcache CVE risk acknowledgement are required"
        )
    venv = root / ".venv"
    site_dir = venv / "lib/python3.12/site-packages"
    activated_dir = venv / RUNTIME_FOLDER
    activation = site_dir / PTH_FILE
    lock = venv / ".ruth-native-install.lock"
    if site_dir.is_symlink() or not site_dir.is_dir() \
            or activated_dir.is_symlink() or activated_dir.exists() \
            or activation.is_symlink() or activation.exists():
        return _result(
            "occupied_or_unsafe_venv_location",
            venv=True, pins=True, wheel=True, risk=True,
            checksum=initial.wheel_sha256,
            error="Installation paths already exist or are symlinks"
        )
    try:
        lock.mkdir(mode=0o700)
    except (OSError, FileExistsError):
        return _result("install_busy", venv=True, pins=True, wheel=True,
                       risk=True, checksum=initial.wheel_sha256)
    stage = None
    activated = False
    started = False
    target_moved = False
    try:
        if activated_dir.exists() or activation.exists():
            return _result("occupied_or_unsafe_venv_location",
                           venv=True, pins=True, wheel=True, risk=True)
        work = root / BUILD_ROOT
        if work.is_symlink():
            return _result("unsafe_build_root",
                           venv=True, pins=True, wheel=True, risk=True)
        stage = Path(tempfile.mkdtemp(dir=work, prefix=".native-install-stage-"))
        target = stage / "packages"
        target.mkdir(mode=0o700)
        # Offline installation into TEMPORARY TARGET; does not touch live venv.
        command = [
            str(venv / "bin/python"), "-m", "pip", "--isolated", "install",
            "--no-index", "--no-deps", "--no-cache-dir", "--no-compile",
            "--no-input", "--target", str(target),
            *(str(path) for path in wheels),
        ]
        started = True
        result = run(
            command, cwd=str(root), capture_output=True, text=True,
            check=False, timeout=600,
            env={**os.environ, "PIP_NO_INDEX": "1",
                 "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull},
        )
        if result.returncode:
            return _result("staging_install_failed",
                           venv=True, pins=True, wheel=True, risk=True,
                           started=True, checksum=initial.wheel_sha256,
                           error=f"Offline target pip exit {result.returncode}")
        if not probe(venv / "bin/python", target, staged=True,
                     run=run, root=root):
            return _result(
                "staged_cuda_probe_failed", venv=True, pins=True,
                wheel=True, risk=True, started=True,
                checksum=initial.wheel_sha256,
                error="Staged binding import or CUDA support test failed; venv unchanged"
            )
        # Two filesystem mutations to the live venv. Roll back on any failure.
        target.rename(activated_dir)
        target_moved = True
        # A .pth file contains only a fixed local directory, never executable code.
        tmp_pth = site_dir / (PTH_FILE + ".tmp")
        try:
            with tmp_pth.open("x", encoding="utf-8") as handle:
                handle.write(str(activated_dir.resolve()) + "\n")
            os.replace(tmp_pth, activation)
            activated = True
        finally:
            if tmp_pth.exists():
                tmp_pth.unlink()
        if not probe(venv / "bin/python", activated_dir, staged=False,
                     run=run, root=root):
            raise RuntimeError("Activated environment failed CUDA import test")
        return _result(
            "native_cuda_binding_installed", venv=True, pins=True,
            wheel=True, risk=True, started=True, installed=True,
            supported=True, checksum=initial.wheel_sha256,
            location=activated_dir,
        )
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        rolled_back = bool(target_moved or activated)
        if activated:
            activation.unlink(missing_ok=True)
        if target_moved and activated_dir.exists():
            shutil.rmtree(activated_dir)
        return _result(
            "installation_rolled_back", venv=True, pins=True,
            wheel=True, risk=True, started=started,
            checksum=initial.wheel_sha256, rolled_back=rolled_back,
            error=type(exc).__name__ + ": " + str(exc)[:220]
        )
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)
        lock.rmdir()
