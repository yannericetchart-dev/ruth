"""NVIDIA-only GPU discovery through the installed nvidia-smi utility."""
from .runtime import detect

__all__ = ["detect"]
