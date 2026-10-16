"""Portable CPU and RAM inventory, without GPU/backend probing."""
from .runtime import DeviceInfo, HardwareReport, ResourceReport, detect, resources

__all__ = ["DeviceInfo", "HardwareReport", "ResourceReport", "detect", "resources"]
