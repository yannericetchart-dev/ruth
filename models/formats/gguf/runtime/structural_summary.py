"""Catalog-bound structural summary of GGUF files, without reading tensor payload.

The earlier INDEX and BLOCKS readers intentionally cannot size all quantized
types. Their "size_unverified" is not an error when a subsequent KBLOCKS
inspection checks the corresponding extents. Unsupported layouts remain
unverified and must NEVER be advertised as structurally checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from models.catalog import ModelRecord, catalog
from .reader import _root, _inspect as _inspect_header
from .metadata import _inspect as _inspect_metadata
from .tensor_index import _inspect as _inspect_index
from .blocks import _inspect as _inspect_blocks
from .kblocks import _inspect as _inspect_kblocks


@dataclass(frozen=True)
class GGUFStructuralSummary:
    """Portable per-model result: no paths, metadata values or tensor names."""

    model_id: str
    status: str  # structural_checked / size_unverified / invalid
    reason: str = ""
    tensor_count: int = 0
    size_unverified: int = 0


@dataclass(frozen=True)
class GGUFStructuralReport:
    """Aggregate of declared GGUF; no model discovery or mutation."""

    status: str  # structural_checked / size_unverified / invalid / not_configured
    models: tuple[GGUFStructuralSummary, ...] = ()
    reason: str = ""

    @property
    def count(self) -> int:
        return len(self.models)


def _inspect(record: ModelRecord, root: Path) -> GGUFStructuralSummary:
    """Reuse already bounded readers on one CAT-vetted record."""
    if record.status != "present_unverified":
        return GGUFStructuralSummary(record.id, "invalid", "CAT:missing")

    header = _inspect_header(record, root)
    if header.status != "header_read":
        return GGUFStructuralSummary(record.id, "invalid", "GGUF:" + header.status)

    metadata = _inspect_metadata(record, root)
    if metadata.status != "metadata_read":
        return GGUFStructuralSummary(record.id, "invalid", "META:" + metadata.status)

    index = _inspect_index(record, root)
    if index.status not in ("index_checked", "size_unverified"):
        return GGUFStructuralSummary(record.id, "invalid", "INDEX:" + index.status)

    blocks = _inspect_blocks(record, root)
    if blocks.status not in ("layout_checked", "size_unverified"):
        return GGUFStructuralSummary(record.id, "invalid", "BLOCKS:" + blocks.status)

    kblocks = _inspect_kblocks(record, root)
    if kblocks.status not in ("kblocks_checked", "size_unverified"):
        return GGUFStructuralSummary(record.id, "invalid", "KBLOCKS:" + kblocks.status)

    if kblocks.tensor_count != index.tensor_count or blocks.tensor_count != index.tensor_count:
        return GGUFStructuralSummary(record.id, "invalid", "INDEX:inconsistent_count")

    if kblocks.status == "size_unverified":
        return GGUFStructuralSummary(
            record.id, "size_unverified", "KBLOCKS:size_unverified",
            kblocks.tensor_count, kblocks.size_unverified,
        )

    return GGUFStructuralSummary(record.id, "structural_checked", "",
                                 kblocks.tensor_count, 0)


def inspect_structural_summaries(*, project_root: Path | None = None) -> GGUFStructuralReport:
    """Summarize only explicitly declared GGUF, never read/model tensor bytes."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status == "not_configured":
        return GGUFStructuralReport("not_configured", reason="CAT:not_configured")
    if declared.status == "invalid":
        return GGUFStructuralReport("invalid", reason="CAT:invalid_catalog")
    records = tuple(item for item in declared.models if item.format == "gguf")
    if not records:
        return GGUFStructuralReport("not_configured", reason="CAT:no_gguf")

    models = tuple(_inspect(record, root) for record in records)
    if any(row.status == "invalid" for row in models):
        status = "invalid"
    elif any(row.status == "size_unverified" for row in models):
        status = "size_unverified"
    else:
        status = "structural_checked"
    return GGUFStructuralReport(status, models)
