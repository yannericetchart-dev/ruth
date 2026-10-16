"""Immutable and portable GGUF header-only results."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GGUFHeader:
    """A declared model's first 24 bytes, not a whole-file integrity check."""

    model_id: str
    status: str  # header_read / missing / truncated_header / invalid_magic / ...
    version: int | None = None
    tensor_count: int | None = None
    metadata_kv_count: int | None = None
    byte_order: str | None = None  # little / big; only for supported headers


@dataclass(frozen=True)
class GGUFReport:
    """One transient result per declared GGUF file; never persist host data."""

    status: str  # not_configured / empty / no_gguf / inspected / partial / invalid
    headers: tuple[GGUFHeader, ...]
    errors: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.headers)
