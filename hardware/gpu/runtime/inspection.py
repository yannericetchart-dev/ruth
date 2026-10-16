"""Aggregate only authorized vendor providers (NVIDIA for GPU-SCOPE-001)."""
from hardware.gpu.nvidia import detect as detect_nvidia
from .contracts import GPUReport


def detect() -> GPUReport:
    """No AMD or Intel discovery until separately authorized."""
    return detect_nvidia()
