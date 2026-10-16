"""Portable CPU provider for HW-001; no GPU or backend detection."""
from .runtime import detect

__all__ = ["detect"]
