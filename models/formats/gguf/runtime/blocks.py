"""Read-only bounded GGML block layout verification for catalogued GGUF.

This checks only tensor descriptors and file offsets. Tensor bytes are never
read, mapped, hashed or decoded. Unsupported layouts remain unverified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.catalog import ModelRecord, catalog
from .reader import _root
from .tensor_index import _inspect as _inspect_index


@dataclass(frozen=True)
class GGUFBlockLayout:
    model_id: str
    status: str
    tensor_count: int = 0
    size_unverified: int = 0


@dataclass(frozen=True)
class GGUFBlockLayoutReport:
    status: str  # not_configured / empty / no_gguf / inspected / partial / invalid
    models: tuple[GGUFBlockLayout, ...]
    errors: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.models)


def _inspect(record: ModelRecord, root: Path) -> GGUFBlockLayout:
    result = _inspect_index(record, root, validate_blocks=True)
    state = "layout_checked" if result.status == "index_checked" else result.status
    return GGUFBlockLayout(result.model_id, state, result.tensor_count, result.size_unverified)


def inspect_block_layout(model_id: str, *, project_root: Path | None = None) -> GGUFBlockLayout:
    """Inspect only a CAT-declared ID, never a caller-supplied file path."""
    root = _root(project_root)
    declaration = catalog(project_root=root)
    if declaration.status == "invalid":
        return GGUFBlockLayout(model_id, "invalid_catalog")
    row = next((entry for entry in declaration.models if entry.id == model_id), None)
    return GGUFBlockLayout(model_id, "not_declared") if row is None else _inspect(row, root)


def inspect_all_block_layouts(*, project_root: Path | None = None) -> GGUFBlockLayoutReport:
    """Inspect declared GGUF records; absent catalog is a nonblocking skip."""
    root = _root(project_root)
    declaration = catalog(project_root=root)
    if declaration.status in ("not_configured", "empty"):
        return GGUFBlockLayoutReport(declaration.status, ())
    if declaration.status == "invalid":
        return GGUFBlockLayoutReport("invalid", (), ("invalid_catalog",))
    rows = tuple(_inspect(entry, root) for entry in declaration.models if entry.format == "gguf")
    if not rows:
        return GGUFBlockLayoutReport("no_gguf", ())
    status = "inspected" if all(entry.status == "layout_checked" for entry in rows) else "partial"
    return GGUFBlockLayoutReport(status, rows)
