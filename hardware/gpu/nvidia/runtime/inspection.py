"""NVIDIA-only GPU inventory from an optional read-only nvidia-smi query."""
from __future__ import annotations

import re
import subprocess

from hardware.gpu.runtime.contracts import GPUDevice, GPUReport

_QUERY = [
    "nvidia-smi", "--query-gpu=name,memory.total",
    "--format=csv,noheader,nounits",
]
_MAX_OUTPUT_CHARS = 65536
_MAX_DEVICES = 32
_TIMEOUT_SECONDS = 5
_SAFE_MODEL = re.compile(r"^[\w][\w .+()\-]{0,159}$", re.UNICODE)


def _parse(stdout: str) -> GPUReport:
    """Only expected CSV fields; no echoing unexpected or identifying data."""
    if not isinstance(stdout, str) or len(stdout) > _MAX_OUTPUT_CHARS:
        return GPUReport((), "unknown")
    rows = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not rows:
        # Successful empty query means *not reported*, never 'no GPU in the PC'.
        return GPUReport((), "none_reported")
    if len(rows) > _MAX_DEVICES:
        return GPUReport((), "unknown")
    found: list[GPUDevice] = []
    invalid = False
    for row in rows:
        if "," not in row:
            invalid = True
            continue
        model_raw, memory_raw = row.rsplit(",", 1)
        model_raw = model_raw.strip()
        memory_raw = memory_raw.strip()
        model = model_raw if _SAFE_MODEL.fullmatch(model_raw) else None
        try:
            memory_mib = int(memory_raw)
            memory = memory_mib * 1024 * 1024 if memory_mib > 0 else None
        except (ValueError, OverflowError):
            memory = None
        if model is None and memory is None:
            invalid = True
            continue
        found.append(GPUDevice(
            vendor="NVIDIA",
            model=model,
            total_vram_bytes=memory,
            status="available" if model is not None and memory is not None else "partial",
        ))
    if not found:
        return GPUReport((), "unknown")
    status = "partial" if invalid or any(x.status == "partial" for x in found) else "available"
    return GPUReport(tuple(found), status)


def detect() -> GPUReport:
    """Query nvidia-smi with fixed arguments, no shell or driver installation.

    Errors (including stderr and raw command outputs) are deliberately not
    retained or printed: DATA-001 prohibits sharing machine-specific logs.
    """
    try:
        result = subprocess.run(
            _QUERY,
            shell=False,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return GPUReport((), "unavailable")
    except subprocess.TimeoutExpired:
        return GPUReport((), "timeout")
    except (OSError, ValueError, UnicodeError):
        return GPUReport((), "inspection_failed")
    if result.returncode != 0:
        return GPUReport((), "inspection_failed")
    return _parse(result.stdout)
