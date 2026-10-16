"""Read-only GPU inventory contracts, separate from CPU/physical RAM."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GPUDevice:
    """Only descriptive fields; no UUID, serial number or persistent ID."""

    vendor: str
    model: str | None
    total_vram_bytes: int | None
    status: str  # available / partial


@dataclass(frozen=True)
class GPUReport:
    """Observed provider result, not proof that a GPU is absent or CUDA-ready."""

    devices: tuple[GPUDevice, ...]
    status: str  # available/partial/none_reported/unavailable/timeout/inspection_failed/unknown

    @property
    def count(self) -> int:
        return len(self.devices)

    def to_dict(self) -> dict:
        """Transient only: never publish actual device inventories to Git."""
        return asdict(self)
