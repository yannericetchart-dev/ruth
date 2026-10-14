"""GNU/Linux x86_64 hardware inventory (phase 01), without inference probes.

Generic DRM PCI detection covers NVIDIA, AMD and Intel. Only NVIDIA telemetry
is implemented (nvidia-smi), matched by PCI address. Other GPU telemetry: TO DO.
No optional inference libraries are imported or initialized.
Run: python -m ruth_runtime.hardware
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Callable, Sequence

from .contracts import DeviceInfo

_PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
_PCI_PATTERN = re.compile(r"^(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2}\.[0-7])$")
_CARD_PATTERN = re.compile(r"card[0-9]+$")
_MIB = 1024 * 1024


@dataclass(frozen=True)
class HostCPU:
    architecture: str
    vendor: str
    model: str | None
    logical_processors: int | None
    physical_cores: int | None
    flags: tuple[str, ...]


@dataclass(frozen=True)
class HostMemory:
    total_bytes: int | None
    available_bytes: int | None
    swap_total_bytes: int | None


@dataclass(frozen=True)
class HardwareReport:
    os_name: str
    os_release: str
    architecture: str
    supported_platform: bool
    cpu: HostCPU
    memory: HostMemory
    gpus: tuple[DeviceInfo, ...]
    cuda_visible_devices: str | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        # Presence of NVIDIA hardware is NOT CUDA/runtime verification.
        return asdict(self)


def normalize_pci_address(address: str | None) -> str | None:
    """Normalizes standard PCI BDF and NVIDIA's eight-digit domain."""
    if not address:
        return None
    match = _PCI_PATTERN.fullmatch(address.strip())
    if not match:
        return None
    domain = match.group("domain")
    if len(domain) == 8:
        if domain[:4].lower() != "0000":
            return None
        domain = domain[4:]
    return (domain + ":" + match.group("bus") + ":" + match.group("device")).lower()


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return None


def parse_meminfo(text: str) -> HostMemory:
    """Linux /proc/meminfo uses KiB. Unknown data stays None."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"([A-Za-z_()]+):\s*([0-9]+)\s+kB\s*", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return HostMemory(values.get("MemTotal"), values.get("MemAvailable"),
                      values.get("SwapTotal"))


def parse_cpuinfo(text: str, *, arch: str, logical: int | None = None) -> HostCPU:
    records: list[dict[str, str]] = []
    for part in re.split(r"\n\s*\n", text.strip()):
        data = dict(line.split(":", 1) for line in part.splitlines() if ":" in line)
        record = {k.strip(): v.strip() for k, v in data.items()}
        if record:
            records.append(record)
    first = records[0] if records else {}
    maker = first.get("vendor_id", "").lower()
    vendor = {"genuineintel": "intel", "authenticamd": "amd"}.get(
        maker, maker or "unknown"
    )
    pairs = {(r["physical id"], r["core id"]) for r in records
             if "physical id" in r and "core id" in r}
    flags = tuple(sorted(set(first.get("flags", "").split())))
    return HostCPU(arch, vendor, first.get("model name"),
                   logical if logical is not None else (len(records) or None),
                   len(pairs) if pairs else None, flags)


def _sysfs_gpus(drm_root: Path) -> list[DeviceInfo]:
    """PCI inventory via DRM sysfs, independent of vendor libraries."""
    found: dict[str, DeviceInfo] = {}
    try:
        cards = sorted(drm_root.glob("card[0-9]*"))
    except OSError:
        return []
    for card in cards:
        if not _CARD_PATTERN.fullmatch(card.name):
            continue
        root = card / "device"
        code = _read(root / "vendor")
        if not code:
            continue
        vendor = _PCI_VENDORS.get(code.strip().lower(), "other")
        uevent = _read(root / "uevent") or ""
        slot = next((line.split("=", 1)[1] for line in uevent.splitlines()
                     if line.startswith("PCI_SLOT_NAME=")), None)
        pci = normalize_pci_address(slot) or normalize_pci_address(root.resolve().name)
        identifier = "pci:" + pci if pci else "drm:" + card.name
        found.setdefault(identifier, DeviceInfo(
            identifier=identifier, kind="gpu", vendor=vendor,
            name=vendor.upper() + " GPU (DRM)", pci_address=pci,
        ))
    return list(found.values())


@dataclass(frozen=True)
class _NvidiaObservation:
    uuid: str
    pci_address: str | None
    name: str
    total_bytes: int | None
    free_bytes: int | None
    driver: str | None


def _mib_value(value: str) -> int | None:
    try:
        number = int(value.strip())
        return number * _MIB if number >= 0 else None
    except ValueError:
        return None


def parse_nvidia_smi(text: str) -> tuple[_NvidiaObservation, ...]:
    observations: list[_NvidiaObservation] = []
    # NVIDIA GPU names may contain commas; respect CSV quoting and leading spaces.
    for row in csv.reader(io.StringIO(text), skipinitialspace=True):
        if len(row) != 6:
            continue
        uuid, bus, name, total, free, driver = (v.strip() for v in row)
        if not uuid.startswith("GPU-"):
            continue
        observations.append(_NvidiaObservation(
            uuid=uuid, pci_address=normalize_pci_address(bus), name=name,
            total_bytes=_mib_value(total), free_bytes=_mib_value(free),
            driver=driver if driver and driver.upper() != "N/A" else None,
        ))
    return tuple(observations)


def _nvidia_probe(run: Callable[..., object]) -> tuple[tuple[_NvidiaObservation, ...], str | None]:
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,memory.total,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        )
    except FileNotFoundError:
        return (), "nvidia-smi assente: telemetria NVIDIA non disponibile"
    except (OSError, subprocess.SubprocessError) as exc:
        return (), f"nvidia-smi non disponibile ({type(exc).__name__}): telemetria non verificata"
    observations = parse_nvidia_smi(result.stdout)
    if not observations:
        return (), "nvidia-smi non ha restituito record validi; telemetria non verificata"
    return observations, None


def _merge_nvidia(gpus: Sequence[DeviceInfo],
                  observations: Sequence[_NvidiaObservation],
                  warnings: list[str]) -> tuple[DeviceInfo, ...]:
    indexed = {d.pci_address: i for i, d in enumerate(gpus) if d.pci_address}
    merged = list(gpus)
    seen: set[str] = set()
    for item in observations:
        if item.pci_address and item.pci_address in seen:
            warnings.append("Record NVIDIA duplicato per PCI " + item.pci_address)
            continue
        if item.pci_address:
            seen.add(item.pci_address)
        if item.pci_address in indexed:
            i = indexed[item.pci_address]
            if merged[i].vendor != "nvidia":
                warnings.append("Mismatch vendor tra DRM e NVIDIA per " + str(item.pci_address))
                continue
            merged[i] = replace(
                merged[i], name=item.name, uuid=item.uuid, driver=item.driver,
                total_memory_bytes=item.total_bytes,
                available_memory_bytes=item.free_bytes,
            )
        elif item.pci_address:
            # Headless GPU: visible to nvidia-smi but not DRM.
            merged.append(DeviceInfo(
                identifier="pci:" + item.pci_address, kind="gpu",
                vendor="nvidia", name=item.name,
                pci_address=item.pci_address, uuid=item.uuid, driver=item.driver,
                total_memory_bytes=item.total_bytes,
                available_memory_bytes=item.free_bytes,
            ))
        else:
            # A missing PCI identity never justifies merging by guessed order.
            warnings.append("Identità PCI assente per " + item.uuid + "; nessun merge inferito")
    return tuple(sorted(merged, key=lambda d: d.identifier))


def inspect_linux_hardware(*, proc_root: Path = Path("/proc"),
                           drm_root: Path = Path("/sys/class/drm"),
                           run: Callable[..., object] = subprocess.run,
                           system: str | None = None,
                           architecture: str | None = None,
                           release: str | None = None,
                           logical_processors: int | None = None) -> HardwareReport:
    """Read-only inventory. Never load a model or initialize CUDA."""
    system = platform.system() if system is None else system
    architecture = platform.machine() if architecture is None else architecture
    release = platform.release() if release is None else release
    supported = system == "Linux" and architecture == "x86_64"
    warnings: list[str] = []
    if not supported:
        warnings.append("Piattaforma fuori ambito: richiesti GNU/Linux x86_64")
    cpu_text = _read(proc_root / "cpuinfo") if system == "Linux" else None
    mem_text = _read(proc_root / "meminfo") if system == "Linux" else None
    if cpu_text is None:
        warnings.append("/proc/cpuinfo non leggibile: CPU parzialmente identificata")
    if mem_text is None:
        warnings.append("/proc/meminfo non leggibile: RAM sconosciuta")
    cpu = parse_cpuinfo(cpu_text or "", arch=architecture,
                        logical=logical_processors)
    memory = parse_meminfo(mem_text or "")
    gpus: tuple[DeviceInfo, ...] = ()
    if supported:
        drm = _sysfs_gpus(drm_root)
        observed, warning = _nvidia_probe(run)
        if warning and any(d.vendor == "nvidia" for d in drm):
            warnings.append(warning)
        elif warning and not drm and warning != "nvidia-smi assente: telemetria NVIDIA non disponibile":
            warnings.append(warning)
        gpus = _merge_nvidia(drm, observed, warnings)
    return HardwareReport(
        os_name=system, os_release=release, architecture=architecture,
        supported_platform=supported, cpu=cpu, memory=memory,
        gpus=gpus, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        warnings=tuple(warnings),
    )


class LinuxHardwareProvider:
    """Generic HardwareProvider with a detailed read-only inventory."""

    def inspect(self) -> HardwareReport:
        return inspect_linux_hardware()

    def discover(self) -> Sequence[DeviceInfo]:
        report = self.inspect()
        return (DeviceInfo(
            identifier="cpu:host", kind="cpu", vendor=report.cpu.vendor,
            name=report.cpu.model or "CPU x86_64",
        ), *report.gpus)


def main() -> int:
    report = inspect_linux_hardware()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True,
                     ensure_ascii=False))
    return 0 if report.supported_platform else 2


if __name__ == "__main__":
    raise SystemExit(main())
