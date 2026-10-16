"""Portable, transient CPU and RAM inventory contracts (HW-SCOPE-001)."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DeviceInfo:
    """CPU only in HW-001. GPU inventory is a separate future component."""

    kind: str  # cpu
    architecture: str | None
    logical_processors: int | None
    status: str  # available / partial / unknown


@dataclass(frozen=True)
class ResourceReport:
    """Total physical memory as exposed by the OS; not usable/free RAM."""

    total_memory_bytes: int | None
    status: str  # available / unknown

    def to_dict(self) -> dict:
        """Transient data: real machine inventories must remain local."""
        return asdict(self)


@dataclass(frozen=True)
class HardwareReport:
    """Read-only inventory, never a certificate of inference readiness."""

    devices: tuple[DeviceInfo, ...]
    memory: ResourceReport
    status: str  # available / partial / unknown

    def to_dict(self) -> dict:
        """Transient data: never version observed inventories (DATA-001)."""
        return asdict(self)
