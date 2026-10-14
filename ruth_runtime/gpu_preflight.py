"""Read-only NVIDIA GPU exclusivity and real free-VRAM preflight.

Never kills processes, changes display-manager services, initializes CUDA or
assumes nominal VRAM equals free VRAM. Fail closed when process telemetry cannot
be established. nvidia-smi XML includes graphics and compute processes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
from typing import Any, Callable
from xml.etree import ElementTree

_MIB = re.compile(r"^([0-9]+) MiB$")
_UUID = re.compile(r"^GPU-[A-Za-z0-9-]{12,64}$")


@dataclass(frozen=True)
class GpuPreflight:
    status: str
    gpu_uuid: str
    driver_version: str | None
    vram_class_gib: int | None
    memory_mib: dict[str, int | None] | None
    processes: tuple[dict[str, Any], ...]
    recommendations: tuple[str, ...]
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _mib(raw: str | None, *, required: bool = False) -> int | None:
    value = (raw or "").strip()
    if value in ("", "N/A", "[Not Supported]") and not required:
        return None
    match = _MIB.fullmatch(value)
    if not match:
        raise ValueError("NVIDIA memory telemetry unavailable or malformed")
    return int(match.group(1))


def _advice(processes: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    advice = [
        "Chiudere sul sistema host i processi GPU concorrenti; Ruth non termina processi automaticamente.",
        "Verificare nuovamente con nvidia-smi e sudo fuser -v /dev/nvidia* prima del probe.",
    ]
    if any(Path(p["name"]).name.lower() in {"xorg", "x"} for p in processes):
        advice.insert(1,
            "Xorg usa la GPU NVIDIA: salvare il lavoro; da TTY o SSH si puo' "
            "valutare sudo systemctl stop display-manager, poi "
            "sudo systemctl start display-manager per ripristinarlo. "
            "Questo arresta la sessione grafica; non eseguirlo dal suo terminale.")
    return tuple(advice)


def parse_gpu_xml(xml: str, gpu_uuid: str) -> GpuPreflight:
    """Parse an nvidia-smi -q -x snapshot for exactly the selected GPU UUID."""
    if not _UUID.fullmatch(gpu_uuid) or len(xml) > 2_000_000 or "<!ENTITY" in xml:
        raise ValueError("Invalid selected GPU or unsafe NVIDIA XML")
    root = ElementTree.fromstring(xml)
    if root.tag != "nvidia_smi_log":
        raise ValueError("Unexpected NVIDIA XML root")
    matches = [gpu for gpu in root.findall("gpu")
               if (gpu.findtext("uuid") or "").strip() == gpu_uuid]
    if len(matches) != 1:
        raise ValueError("Selected NVIDIA UUID missing or ambiguous in XML")
    gpu = matches[0]
    fb = gpu.find("fb_memory_usage")
    if fb is None:
        raise ValueError("Missing framebuffer memory telemetry")
    memory = {key: _mib(fb.findtext(key), required=key in ("total", "free"))
              for key in ("total", "reserved", "used", "free")}
    if memory["total"] <= 0 or memory["free"] > memory["total"]:
        raise ValueError("Inconsistent NVIDIA framebuffer memory telemetry")
    process_node = gpu.find("processes")
    if process_node is None:
        raise ValueError("NVIDIA process telemetry not available")
    processes = []
    for p in process_node.findall("process_info"):
        raw_pid = (p.findtext("pid") or "").strip()
        kind = (p.findtext("type") or "").strip()
        name = (p.findtext("process_name") or "").strip()
        if not raw_pid.isdecimal() or int(raw_pid) <= 0 or not kind or not name:
            raise ValueError("Incomplete NVIDIA process telemetry")
        processes.append({
            "pid": int(raw_pid), "type": kind,
            "name": name[:220], "gpu_memory_mib": _mib(p.findtext("used_memory")),
        })
    if not processes and (process_node.text or "").strip() not in (
        "", "No running processes found"
    ):
        raise ValueError("NVIDIA reports unidentified GPU processes")
    running = tuple(sorted(processes, key=lambda p: p["pid"]))
    driver = (root.findtext("driver_version") or "").strip() or None
    return GpuPreflight(
        status="gpu_busy" if running else "gpu_ready",
        gpu_uuid=gpu_uuid, driver_version=driver,
        vram_class_gib=round(memory["total"] / 1024),
        memory_mib=memory, processes=running,
        recommendations=_advice(running) if running else (),
    )


def inspect_gpu_preflight(
    gpu_uuid: str, *, run: Callable[..., Any] = subprocess.run,
) -> GpuPreflight:
    """Read-only snapshot. A telemetry failure MUST prevent GPU inference."""
    try:
        if not _UUID.fullmatch(gpu_uuid):
            raise ValueError("Invalid selected GPU UUID")
        result = run(
            ["nvidia-smi", "-q", "-x", "-i", gpu_uuid],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return parse_gpu_xml(result.stdout, gpu_uuid)
    except (OSError, subprocess.SubprocessError, ValueError,
            ElementTree.ParseError, TypeError) as exc:
        return GpuPreflight(
            status="gpu_preflight_unverified", gpu_uuid=gpu_uuid,
            driver_version=None, vram_class_gib=None, memory_mib=None,
            processes=(),
            recommendations=(
                "Verificare sul sistema host: nvidia-smi -q -d MEMORY; nvidia-smi; sudo fuser -v /dev/nvidia*.",
                "Non eseguire il probe finche' Ruth non riesce a leggere tutti i processi GPU.",
            ),
            error=type(exc).__name__ + ": " + str(exc)[:220],
        )
