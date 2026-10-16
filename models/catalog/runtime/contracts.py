"""Portable, read-only local model catalog contracts for CAT-SCOPE-001."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelRecord:
    """Declared model: file presence is not format or runtime verification."""

    id: str
    format: str
    reference: str  # Relative to .local/models, not an absolute host path.
    status: str  # present_unverified / missing


@dataclass(frozen=True)
class ModelCatalog:
    """Local-only catalog result; no network, download or directory scan."""

    status: str  # not_configured / empty / ready / partial / invalid
    models: tuple[ModelRecord, ...]
    errors: tuple[str, ...]  # Portable reason codes, never raw file content.

    @property
    def count(self) -> int:
        return len(self.models)

    def to_dict(self) -> dict:
        """Transient: never commit actual local model references (DATA-001)."""
        return asdict(self)
