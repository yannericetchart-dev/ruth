"""Combine only CPU inventory and optionally available physical RAM."""
import os

from hardware.cpu import detect as detect_cpu
from .contracts import DeviceInfo, HardwareReport, ResourceReport


def resources(device: DeviceInfo | str = "system") -> ResourceReport:
    """Read total physical RAM if portable OS sysconf keys are supported.

    Returns unknown rather than guessing or launching native utilities.
    Does not measure available RAM, cgroup quotas, or GPU memory.
    """
    if device != "system" and not (isinstance(device, DeviceInfo) and device.kind == "cpu"):
        raise ValueError("Risorse supportate: sistema o dispositivo CPU.")
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError, TypeError, OverflowError):
        return ResourceReport(None, "unknown")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (page_size, pages)):
        return ResourceReport(None, "unknown")
    return ResourceReport(page_size * pages, "available")


def detect() -> HardwareReport:
    """CPU and RAM only; this is not a compatibility or GPU inspection."""
    devices = detect_cpu()
    ram = resources("system")
    cpu_status = devices[0].status
    if cpu_status == "available" and ram.status == "available":
        status = "available"
    elif cpu_status == "unknown" and ram.status == "unknown":
        status = "unknown"
    else:
        status = "partial"
    return HardwareReport(tuple(devices), ram, status)
