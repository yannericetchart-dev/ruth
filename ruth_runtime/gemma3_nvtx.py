"""Optional NVTX phase markers for external CUDA dispatch traces.

NVTX availability is not evidence that a CUDA kernel was dispatched.
No GPU initialization or inference occurs here. Missing NVTX is safe.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
from ctypes.util import find_library


class NativeRanges:
    def __init__(self, experiment: str = "PHASE27"):
        if experiment not in ("PHASE27", "PHASE28"):
            raise ValueError("Unapproved NVTX experiment")
        self.experiment = experiment
        self._lib = None
        try:
            name = find_library("nvToolsExt")
            if name:
                lib = ctypes.CDLL(name)
                lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
                lib.nvtxRangePushA.restype = ctypes.c_int
                lib.nvtxRangePop.argtypes = []
                lib.nvtxRangePop.restype = ctypes.c_int
                self._lib = lib
        except (OSError, ValueError, AttributeError):
            self._lib = None

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
