"""Initial NVIDIA/CUDA runtime policy, independent of host detection.

Ruth targets the NVIDIA R595 driver family and CUDA 13.2 as its minimum
baseline. CUDA 13.2 GA on Linux requires >=595.45.04 for full feature
compatibility (NVIDIA CUDA 13.2 release notes). CUDA minor compatibility with
older drivers does not override this deliberately stricter Ruth policy.

https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/
"""
from __future__ import annotations

import re

MIN_NVIDIA_DRIVER = (595, 45, 4)
MIN_CUDA_VERSION = (13, 2)

_DRIVER_RE = re.compile(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?\s*$")
_CUDA_RE = re.compile(r"\bCUDA Version:\s*(\d+)\.(\d+)\b")


def parse_driver_version(version: str | None) -> tuple[int, int, int] | None:
    if not isinstance(version, str):
        return None
    match = _DRIVER_RE.fullmatch(version)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def driver_meets_policy(version: str | None) -> bool:
    parsed = parse_driver_version(version)
    return parsed is not None and parsed >= MIN_NVIDIA_DRIVER


def parse_reported_cuda_version(output: str) -> tuple[int, int] | None:
    """CUDA Version in nvidia-smi is driver-supported API, NOT installed Toolkit."""
    match = _CUDA_RE.search(output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def cuda_meets_policy(version: tuple[int, int] | None) -> bool:
    return version is not None and version >= MIN_CUDA_VERSION


def render_version(parts: tuple[int, ...]) -> str:
    if len(parts) == 3:
        return f"{parts[0]}.{parts[1]}.{parts[2]:02d}"
    return ".".join(str(v) for v in parts)
