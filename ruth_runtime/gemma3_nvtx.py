"""Optional NVTX phase markers for external CUDA dispatch traces.

NVTX availability is not evidence that a CUDA kernel was dispatched.
No GPU initialization or inference occurs here. Missing NVTX is safe.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
from ctypes.util import find_library
import os
from pathlib import Path

from .dependencies import PROJECT_ROOT


class NativeRanges:
    def __init__(self, experiment: str = "PHASE27"):
        if experiment not in ("PHASE27", "PHASE28"):
            raise ValueError("Unapproved NVTX experiment")
        self.experiment = experiment
        self._lib = None
        # Phase28 only: explicit opt-in to a user-built, local, NVTX3 API
        # bridge. Never compile/download/load an unrequested native module.
        requested = os.environ.get("RUTH_PHASE28_NVTX3_BRIDGE")
        if experiment == "PHASE28" and requested is not None:
            if requested != "1":
                raise RuntimeError("RUTH_PHASE28_NVTX3_BRIDGE must be 1")
            base = PROJECT_ROOT / ".ruth-build" / "phase28"
            file = base / "libruth_phase28_nvtx3.so"
            if (not base.is_dir() or base.is_symlink()
                or not file.is_file() or file.is_symlink()
                or file.resolve().parent != base.resolve()):
                raise RuntimeError("Explicit Phase28 NVTX3 bridge is unavailable")
            try:
                lib = ctypes.CDLL(str(file))
                self._set_prototypes(lib)
            except (OSError, ValueError, AttributeError) as exc:
                raise RuntimeError("Unable to load explicit Phase28 NVTX3 bridge") from exc
            self._lib = lib
            return
        try:
            name = find_library("nvToolsExt")
            if name:
                lib = ctypes.CDLL(name)
                self._set_prototypes(lib)
                self._lib = lib
        except (OSError, ValueError, AttributeError):
            self._lib = None

    @staticmethod
    def _set_prototypes(lib) -> None:
        lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        lib.nvtxRangePushA.restype = ctypes.c_int
        lib.nvtxRangePop.argtypes = []
        lib.nvtxRangePop.restype = ctypes.c_int

    @property
    def available(self) -> bool:
        return self._lib is not None

    @contextmanager
    def range(self, phase: str):
        if phase not in ("prefill", "decode"):
            raise ValueError("Unexpected NVTX benchmark phase")
        if self._lib is not None:
            self._lib.nvtxRangePushA(("RUTH_GEMMA3_" + self.experiment + "_" + phase.upper()).encode("ascii"))
        try:
            yield
        finally:
            if self._lib is not None:
                self._lib.nvtxRangePop()
