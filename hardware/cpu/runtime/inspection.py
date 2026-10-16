"""Portable, safe CPU observation without any third-party packages."""
import os
import platform

from hardware.runtime.contracts import DeviceInfo


def detect() -> tuple[DeviceInfo, ...]:
    """Describe CPU architecture and logical count when observable."""
    try:
        raw_arch = platform.machine()
        arch = raw_arch.strip() if isinstance(raw_arch, str) else None
        arch = arch or None
    except (OSError, RuntimeError, ValueError, AttributeError):
        arch = None
    try:
        raw_count = os.cpu_count()
        count = raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count > 0 else None
    except (OSError, RuntimeError, ValueError, AttributeError):
        count = None
    if arch and count:
        status = "available"
    elif arch or count:
        status = "partial"
    else:
        status = "unknown"
    return (DeviceInfo("cpu", arch, count, status),)
