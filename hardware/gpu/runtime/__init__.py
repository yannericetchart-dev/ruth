"""GPU provider interface and transient inventory contracts."""
from .contracts import GPUDevice, GPUReport
from .inspection import detect

__all__ = ["GPUDevice", "GPUReport", "detect"]
