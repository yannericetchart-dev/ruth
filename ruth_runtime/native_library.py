"""Qualify the exact installed llama.cpp CUDA shared libraries for dual use.

Read-only validation of pinned native wheel vs installed ELF bytes. Tests
direct C ABI and Python binding in fresh isolated subprocesses. No model is
loaded, no native build, no pip and no inference is performed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
from zipfile import ZipFile, BadZipFile

from .dependencies import PROJECT_ROOT
from .install_plan import SOURCE_COMMIT, LLAMA_CPP_SUBMODULE_COMMIT
from .native_builder import BUILD_ROOT, BUILD_OUTPUT, inspect_cuda_wheel
from .native_installer import NATIVE_LOCK, PTH_FILE, RUNTIME_FOLDER
from .venv_packages import EnvironmentReport, inspect_python_environment

_DIRECT_PROBE = r'''
import ctypes, json, os, pathlib, sys
directory = pathlib.Path(sys.argv[1]).resolve(strict=True)
library = ctypes.CDLL(str(directory / "libllama.so"), mode=os.RTLD_NOW | os.RTLD_LOCAL)
support = library.llama_supports_gpu_offload
support.argtypes = []
support.restype = ctypes.c_bool
info = library.llama_print_system_info
info.argtypes = []
info.restype = ctypes.c_char_p
init = library.llama_backend_init
init.argtypes = []
init.restype = None
free = library.llama_backend_free
free.argtypes = []
free.restype = None
gpu = bool(support())
init()
try:
    system_info = (info() or b"").decode("utf-8", errors="replace")[:512]
finally:
    free()
print(json.dumps({"cuda_supported": gpu, "backend_initialized": True,
                  "system_info": system_info, "libllama": str(directory / "libllama.so")}))
'''
_PYTHON_PROBE = r'''
import json, pathlib, sys
import llama_cpp
from llama_cpp import llama_cpp as native
expected = pathlib.Path(sys.argv[1]).resolve(strict=True)
actual = pathlib.Path(native._lib._name).resolve(strict=True)
gpu = bool(llama_cpp.llama_supports_gpu_offload())
print(json.dumps({"version": llama_cpp.__version__,
                  "cuda_supported": gpu, "actual_library": str(actual),
                  "same_libllama": actual == expected}))
'''


@dataclass(frozen=True)
class NativeLibraryReport:
    status: str
    interpreter_in_project_venv: bool
    installed_binding_verified: bool
    wheel_provenance_verified: bool
    shared_libraries_verified: bool
    library_names: tuple[str, ...]
    direct_abi_verified: bool
    python_binding_same_library: bool
    cuda_support_direct: bool
    cuda_support_python: bool
    native_backend_initialized: bool
    model_loaded: bool
    gpu_offload_executed: bool
    inference_executed: bool
    compilation_started: bool
    installation_started: bool
    cpu_fallback_selected: bool
    diskcache_todo_open: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _report(
    status: str, *, venv: bool = False, installed: bool = False,
    provenance: bool = False, libs: bool = False,
    names: tuple[str, ...] = (), direct: bool = False,
    python_same: bool = False, gpu_direct: bool = False,
    gpu_python: bool = False, initialized: bool = False,
    error: str | None = None,
) -> NativeLibraryReport:
    return NativeLibraryReport(
        status=status, interpreter_in_project_venv=venv,
        installed_binding_verified=installed, wheel_provenance_verified=provenance,
        shared_libraries_verified=libs, library_names=names,
        direct_abi_verified=direct, python_binding_same_library=python_same,
        cuda_support_direct=gpu_direct, cuda_support_python=gpu_python,
        native_backend_initialized=initialized, model_loaded=False,
        gpu_offload_executed=False, inference_executed=False,
        compilation_started=False, installation_started=False,
        cpu_fallback_selected=False, diskcache_todo_open=True, error=error,
    )


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_installed_libraries(
    *, root: Path,
) -> tuple[bool, tuple[str, ...], str | None]:
    """Compare every packaged *.so byte-for-byte to installed native dir."""
    venv = root / ".venv"
    target = venv / RUNTIME_FOLDER
    lib = target / "llama_cpp" / "lib"
    activation = venv / "lib/python3.12/site-packages" / PTH_FILE
    wheelroot = root / BUILD_ROOT / BUILD_OUTPUT
    manifest = root / NATIVE_LOCK
    result_file = wheelroot / "build-result.json"
    paths = (venv, target, lib, activation, root / BUILD_ROOT,
             wheelroot, wheelroot / "wheels", manifest, result_file)
    if any(p.is_symlink() for p in paths) or not target.is_dir() \
            or not lib.is_dir() or not activation.is_file():
        return False, (), "Missing or symlinked installed native location"
    try:
        if activation.read_text(encoding="utf-8") != str(target.resolve()) + "\n":
            return False, (), "Activation path does not match installed Ruth runtime"
        pinned = json.loads(manifest.read_text(encoding="utf-8"))
        built = json.loads(result_file.read_text(encoding="utf-8"))
        if pinned.get("schema") != "ruth-native-wheel-lock-v1" \
                or pinned.get("source_commit") != SOURCE_COMMIT \
                or pinned.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT \
                or pinned.get("python") != "3.12" \
                or pinned.get("cuda_architecture") != "89" \
                or built.get("source_commit") != SOURCE_COMMIT \
                or built.get("llama_cpp_commit") != LLAMA_CPP_SUBMODULE_COMMIT \
                or built.get("wheel_sha256") != pinned.get("wheel_sha256") \
                or built.get("ggml_cuda_library_packaged") is not True:
            return False, (), "Native build manifests disagree with the pinned revision"
        name = pinned["wheel_filename"]
        if not isinstance(name, str) or Path(name).name != name:
            return False, (), "Unsafe wheel filename"
        wheel = wheelroot / "wheels" / name
        if wheel.is_symlink():
            return False, (), "Wheel is a symbolic link"
        cuda_ok, digest, error = inspect_cuda_wheel(wheel)
        if not cuda_ok or digest != pinned.get("wheel_sha256") \
                or built.get("wheel_file") != name:
            return False, (), "Built wheel differs from pinned SHA256: " + str(error)
        with ZipFile(wheel) as archive:
            binaries = {}
            for member in archive.namelist():
                if member.startswith("llama_cpp/lib/") and \
                        (member.endswith(".so") or ".so." in member):
                    basename = Path(member).name
                    if not basename or basename in binaries or \
                            Path(member).parent.as_posix() != "llama_cpp/lib":
                        return False, (), "Unexpected or duplicate shared library"
                    binaries[basename] = _digest_bytes(archive.read(member))
        if not {"libllama.so", "libggml-cuda.so"} <= set(binaries):
            return False, (), "Expected CUDA shared libraries absent from native wheel"
        installed_files = list(lib.iterdir())
        if any(p.is_symlink() for p in installed_files):
            return False, (), "Unexpected symlink in installed native library directory"
        installed_shared = {
            p.name for p in installed_files
            if p.name.endswith(".so") or ".so." in p.name
        }
        if installed_shared != set(binaries):
            return False, (), "Installed shared library set differs from pinned wheel"
        for name, sha in binaries.items():
            file = lib / name
            if not file.is_file() or file.is_symlink():
                return False, (), "Missing installed shared library: " + name
            with file.open("rb") as handle:
                check = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    check.update(chunk)
            if check.hexdigest() != sha:
                return False, (), "Installed library checksum mismatch: " + name
        return True, tuple(sorted(binaries)), None
    except (OSError, ValueError, KeyError, TypeError, BadZipFile,
            UnicodeError, json.JSONDecodeError) as exc:
        return False, (), type(exc).__name__ + ": " + str(exc)[:180]


def _execute_probe(
    python: Path, code: str, library: Path, *, isolated_site: bool,
    run: Callable[..., Any], cwd: Path,
) -> dict:
    command = [str(python), "-I", *(["-S"] if isolated_site else []),
               "-c", code, str(library)]
    environ = {**os.environ, "PYTHONNOUSERSITE": "1"}
    environ.pop("LLAMA_CPP_LIB_PATH", None)
    proc = run(command, capture_output=True, text=True, check=False,
               timeout=90, cwd=str(cwd), env=environ)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Native subprocess failed (exit {proc.returncode}): " +
            (proc.stderr or "")[-250:])
    try:
        obj = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError, TypeError) as exc:
        raise RuntimeError("Malformed isolated probe JSON") from exc
    if not isinstance(obj, dict):
        raise RuntimeError("Malformed isolated probe result")
    return obj


def verify_native_library(
    *, project_root: Path = PROJECT_ROOT,
    environment: EnvironmentReport | None = None,
    run: Callable[..., Any] = subprocess.run,
    library_check: Callable[..., tuple[bool, tuple[str, ...], str | None]] =
        _validate_installed_libraries,
) -> NativeLibraryReport:
    """Validate direct ABI and Python wrapper point to identical pinned libllama."""
    root = Path(project_root)
    env = environment if environment is not None else inspect_python_environment()
    correct_venv = bool(
        env.interpreter_in_project_venv
        and env.python_version.startswith("3.12.")
        and Path(env.project_root).resolve() == root.resolve()
        and Path(env.interpreter).resolve() == (root / ".venv/bin/python").resolve()
        and not (root / ".venv").is_symlink()
    )
    if not correct_venv:
        return _report("blocked_venv", error="Invalid Ruth Python 3.12 interpreter")
    valid, names, error = library_check(root=root)
    if not valid:
        return _report("native_library_integrity_failed", venv=True,
                       error=error)
    lib = root / ".venv" / RUNTIME_FOLDER / "llama_cpp/lib"
    filename = lib / "libllama.so"
    python = root / ".venv/bin/python"
    try:
        direct = _execute_probe(python, _DIRECT_PROBE, lib,
                                isolated_site=True, run=run, cwd=root)
        if not direct.get("backend_initialized") \
                or not direct.get("cuda_supported") \
                or Path(direct.get("libllama", "/")).resolve() != filename.resolve():
            return _report(
                "native_direct_cuda_unavailable", venv=True, installed=True,
                provenance=True, libs=True, names=names,
                error="Direct C ABI failed to initialize CUDA-capable backend")
        linked = _execute_probe(python, _PYTHON_PROBE, filename,
                                isolated_site=False, run=run, cwd=root)
        if linked.get("version") != "0.3.36" or \
                linked.get("cuda_supported") is not True or \
                linked.get("same_libllama") is not True or \
                Path(linked.get("actual_library", "/")).resolve() != filename.resolve():
            return _report(
                "native_python_library_mismatch", venv=True, installed=True,
                provenance=True, libs=True, names=names,
                direct=True, initialized=True, gpu_direct=True,
                error="Python binding does not point to pinned direct C library")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return _report(
            "native_library_probe_failed", venv=True, installed=True,
            provenance=True, libs=True, names=names,
            error=type(exc).__name__ + ": " + str(exc)[:400])
    return _report(
        "native_library_dual_abi_verified", venv=True, installed=True,
        provenance=True, libs=True, names=names, direct=True,
        python_same=True, gpu_direct=True, gpu_python=True,
        initialized=True,
    )
