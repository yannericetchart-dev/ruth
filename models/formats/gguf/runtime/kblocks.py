"""Catalog-bound read-only GGUF K-block structural checks.

Never opens tensor payload for reading. Uses the bounded INDEX parser with
the five previous 32-element layouts plus six GGML K super-block layouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.catalog import ModelRecord, catalog
from .reader import _root
from .tensor_index import _inspect as _inspect_index


@dataclass(frozen=True)
class GGUFKBlockLayout:
    model_id: str
    status: str
    tensor_count: int = 0
    size_unverified: int = 0


@dataclass(frozen=True)
class GGUFKBlockReport:
    status: str  # not_configured / empty / no_gguf / inspected / partial / invalid
    models: tuple[GGUFKBlockLayout, ...]
    errors: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.models)


def _inspect(record: ModelRecord, root: Path) -> GGUFKBlockLayout:
    result = _inspect_index(record, root, validate_kblocks=True)
    return GGUFKBlockLayout(
        result.model_id,
        "kblocks_checked" if result.status == "index_checked" else result.status,
        result.tensor_count,
        result.size_unverified,
    )


def inspect_kblock_layout(model_id: str, *, project_root: Path | None = None) -> GGUFKBlockLayout:
    """Inspect only a CAT-001 model ID, never a caller-supplied path."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status == "invalid":
        return GGUFKBlockLayout(model_id, "invalid_catalog")
    record = next((row for row in declared.models if row.id == model_id), None)
    return GGUFKBlockLayout(model_id, "not_declared") if record is None else _inspect(record, root)


def inspect_all_kblock_layouts(*, project_root: Path | None = None) -> GGUFKBlockReport:
    """Inspect each declared GGUF without discovering, downloading or reading weights."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status in ("not_configured", "empty"):
        return GGUFKBlockReport(declared.status, ())
    if declared.status == "invalid":
        return GGUFKBlockReport("invalid", (), ("invalid_catalog",))
    rows = tuple(_inspect(row, root) for row in declared.models if row.format == "gguf")
    if not rows:
        return GGUFKBlockReport("no_gguf", ())
    state = "inspected" if all(row.status == "kblocks_checked" for row in rows) else "partial"
    return GGUFKBlockReport(state, rows)
