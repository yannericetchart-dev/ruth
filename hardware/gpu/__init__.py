"""GPU discovery: NVIDIA only; vendor-neutral contract, no CUDA tests."""
from .runtime import GPUDevice, GPUReport, detect

__all__ = ["GPUDevice", "GPUReport", "detect"]
