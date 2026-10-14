"""Minimal direct C ABI adapter for pinned llama.cpp CUDA shared runtime.

Separate from CudaLlamaAdapter (high-level llama-cpp-python), but they use
the same installed libllama.so. This only manages native backend lifetime
and diagnostics; C-model/context inference APIs remain an explicit TODO.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .dependencies import PROJECT_ROOT
from .native_installer import RUNTIME_FOLDER
from .native_library import NativeLibraryReport, verify_native_library


class DirectNativeUnavailable(RuntimeError):
    pass


class DirectNativeSession:
    """One direct C API backend session; no exposed ctypes handle."""

    __slots__ = ("_lib", "_closed")

    def __init__(self, library: Any):
        required = ("llama_supports_gpu_offload", "llama_backend_init",
                    "llama_backend_free", "llama_print_system_info")
        try:
            support, initialize, finalize, info = (
                getattr(library, n) for n in required
            )
            support.argtypes = []
            support.restype = ctypes.c_bool
            initialize.argtypes = []
            initialize.restype = None
            finalize.argtypes = []
            finalize.restype = None
            info.argtypes = []
            info.restype = ctypes.c_char_p
            if not support():
                raise DirectNativeUnavailable("Pinned C ABI has no CUDA offload support")
            initialize()
        except (AttributeError, OSError) as exc:
            raise DirectNativeUnavailable("Required pinned C symbols unavailable") from exc
        self._lib = library
        self._closed = False

    def system_info(self) -> str:
        if self._closed:
            raise DirectNativeUnavailable("Native backend session closed")
        data = self._lib.llama_print_system_info()
        if not isinstance(data, (bytes, bytearray)):
            raise DirectNativeUnavailable("Unexpected system_info C ABI result")
        return data.decode("utf-8", errors="replace")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._lib.llama_backend_free()

    def __enter__(self) -> DirectNativeSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class DirectLlamaLibraryAdapter:
    """Calls pinned libc API after a fresh native/Python dual-ABI audit.

    This does not implement model loading, decoding or generation, and
    does not claim model GPU-offload execution.
    """
    project_root: Path = PROJECT_ROOT
    verify: Callable[..., NativeLibraryReport] = verify_native_library
    load_library: Callable[[str], Any] = ctypes.CDLL

    def open(self) -> DirectNativeSession:
        root = Path(self.project_root)
        report = self.verify(project_root=root)
        if report.status != "native_library_dual_abi_verified" or \
                not report.shared_libraries_verified or \
                not report.python_binding_same_library or \
                not report.cuda_support_direct or \
                not report.cuda_support_python:
            raise DirectNativeUnavailable(
                "CUDA shared library was not independently verified: " +
                report.status
            )
        lib = root / ".venv" / RUNTIME_FOLDER / "llama_cpp/lib/libllama.so"
        if lib.is_symlink() or not lib.is_file():
            raise DirectNativeUnavailable("Pinned native libllama.so absent")
        try:
            return DirectNativeSession(self.load_library(str(lib.resolve())))
        except (OSError, ValueError) as exc:
            raise DirectNativeUnavailable("Cannot load verified native library") from exc
