"""Bounded, read-only GGUF tensor-index structural inspection.

Only catalog-declared GGUF files are opened. Reads header, metadata KV and
tensor descriptors, NEVER tensor bytes. Unsupported tensor layouts cannot be
certified and yield size_unverified, not a claim of complete validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from models.catalog import ModelRecord, catalog
from .metadata import _Cursor, _ParseError, _KEY
from .reader import _root, _safe_path

_MAX_TENSORS = 16384
_MAX_DIMENSIONS = 4
_MAX_ELEMENTS = (1 << 63) - 1
_MAX_ALIGNMENT = 1024 * 1024
# Only uncompressed scalar GGML types whose byte count is unambiguous.
# Quantized and future types are not interpreted without block-size rules.
_SCALAR_BYTES = {0: 4, 1: 2, 24: 1, 25: 2, 26: 4, 27: 8, 28: 8, 30: 2}
# Restricted, authoritative GGML types: (elements per block, bytes per block).
# No dequantization and no assumptions about newer or other layouts.
_BLOCK_SIZES = {2: (32, 18), 3: (32, 20), 6: (32, 22),
                7: (32, 24), 8: (32, 34)}


@dataclass(frozen=True)
class GGUFTensorIndex:
    model_id: str
    status: str
    tensor_count: int = 0
    size_unverified: int = 0


@dataclass(frozen=True)
class GGUFTensorIndexReport:
    status: str  # not_configured / empty / no_gguf / inspected / partial / invalid
    models: tuple[GGUFTensorIndex, ...]
    errors: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.models)


def _parse(file, *, validate_blocks: bool = False) -> tuple[int, int]:
    cursor = _Cursor(file)
    raw = cursor.read(24)
    if raw[:4] != b"GGUF":
        raise _ParseError("invalid_magic")
    little, big = int.from_bytes(raw[4:8], "little"), int.from_bytes(raw[4:8], "big")
    if little in (2, 3):
        order = "little"
    elif big == 3:
        order = "big"
    else:
        raise _ParseError("unsupported_version")
    tensor_count = int.from_bytes(raw[8:16], order)
    metadata_count = int.from_bytes(raw[16:24], order)
    if tensor_count > _MAX_TENSORS or metadata_count > 16384:
        raise _ParseError("limit_exceeded")
    alignment = 32
    alignment_seen = False
    for _ in range(metadata_count):
        key_size = cursor.number(8, order)
        if not 1 <= key_size <= 65535:
            raise _ParseError("invalid_metadata")
        try:
            key = cursor.read(key_size).decode("ascii")
        except UnicodeDecodeError:
            raise _ParseError("invalid_metadata") from None
        if not _KEY.fullmatch(key):
            raise _ParseError("invalid_metadata")
        kind = cursor.number(4, order)
        if key == "general.alignment":
            if kind != 4 or alignment_seen:
                raise _ParseError("invalid_alignment")
            alignment_seen = True
            alignment = cursor.number(4, order)
            if not 8 <= alignment <= _MAX_ALIGNMENT or alignment % 8:
                raise _ParseError("invalid_alignment")
        else:
            cursor.value(kind, order)

    descriptors: list[tuple[int, int | None]] = []
    offsets: set[int] = set()
    names: set[str] = set()
    unknown_count = 0
    for _ in range(tensor_count):
        try:
            length = cursor.number(8, order)
            if not 1 <= length <= 64:
                raise _ParseError("invalid_tensor_name")
            try:
                name = cursor.read(length).decode("utf-8")
            except UnicodeDecodeError:
                raise _ParseError("invalid_tensor_name") from None
            if not name or "\x00" in name or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
                raise _ParseError("invalid_tensor_name")
            if name in names:
                raise _ParseError("duplicate_tensor_name")
            names.add(name)
            ndims = cursor.number(4, order)
            if not 1 <= ndims <= _MAX_DIMENSIONS:
                raise _ParseError("invalid_dimensions")
            nelems = 1
            first_dimension = None
            for axis in range(ndims):
                dimension = cursor.number(8, order)
                if not dimension:
                    raise _ParseError("invalid_dimensions")
                if axis == 0:
                    first_dimension = dimension
                if nelems > _MAX_ELEMENTS // dimension:
                    raise _ParseError("limit_exceeded")
                nelems *= dimension
            kind = cursor.number(4, order)
            offset = cursor.number(8, order)
            if offset % alignment:
                raise _ParseError("invalid_tensor_alignment")
            if offset in offsets:
                raise _ParseError("duplicate_tensor_offset")
            offsets.add(offset)
            bytes_per_element = _SCALAR_BYTES.get(kind)
            extent = None if bytes_per_element is None else nelems * bytes_per_element
            if extent is None and validate_blocks and kind in _BLOCK_SIZES:
                block_elements, block_bytes = _BLOCK_SIZES[kind]
                if first_dimension % block_elements:
                    raise _ParseError("invalid_block_dimension")
                extent = (nelems // block_elements) * block_bytes
            if extent is None:
                unknown_count += 1
            descriptors.append((offset, extent))
        except _ParseError as error:
            if error.code == "truncated_metadata":
                raise _ParseError("truncated_index") from None
            raise

    # Absolute data origin is after descriptor table and alignment padding.
    end_of_index = file.tell()
    data_start = end_of_index + (-end_of_index % alignment)
    if tensor_count and data_start >= cursor.size:
        raise _ParseError("tensor_data_missing")
    if not tensor_count and end_of_index > cursor.size:
        raise _ParseError("truncated_index")
    total_bytes = max(0, cursor.size - data_start)
    intervals = sorted(descriptors)
    for idx, (offset, extent) in enumerate(intervals):
        if offset >= total_bytes:
            raise _ParseError("tensor_offset_out_of_file")
        if extent is not None:
            if offset + extent > total_bytes:
                raise _ParseError("tensor_extent_out_of_file")
            if idx + 1 < len(intervals) and offset + extent > intervals[idx + 1][0]:
                raise _ParseError("tensor_overlap")
    return tensor_count, unknown_count


def _inspect(record: ModelRecord, root: Path, *, validate_blocks: bool = False) -> GGUFTensorIndex:
    if record.format != "gguf":
        return GGUFTensorIndex(record.id, "not_gguf")
    if record.status == "missing":
        return GGUFTensorIndex(record.id, "missing")
    path = _safe_path(root, record.reference)
    if path is None:
        return GGUFTensorIndex(record.id, "unsafe_reference")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            count, unknown = _parse(stream, validate_blocks=validate_blocks)
        return GGUFTensorIndex(record.id, "size_unverified" if unknown else "index_checked", count, unknown)
    except _ParseError as error:
        return GGUFTensorIndex(record.id, error.code)
    except (OSError, ValueError, OverflowError):
        return GGUFTensorIndex(record.id, "unreadable")


def inspect_tensor_index(model_id: str, *, project_root: Path | None = None) -> GGUFTensorIndex:
    """Inspect only a model identifier from CAT-001, not a caller-provided path."""
    root = _root(project_root)
    declaration = catalog(project_root=root)
    if declaration.status == "invalid":
        return GGUFTensorIndex(model_id, "invalid_catalog")
    record = next((x for x in declaration.models if x.id == model_id), None)
    return GGUFTensorIndex(model_id, "not_declared") if record is None else _inspect(record, root)


def inspect_all_tensor_indexes(*, project_root: Path | None = None) -> GGUFTensorIndexReport:
    """Inspect all catalogued GGUF tensor indexes without reading tensor contents."""
    root = _root(project_root)
    declaration = catalog(project_root=root)
    if declaration.status in ("not_configured", "empty"):
        return GGUFTensorIndexReport(declaration.status, ())
    if declaration.status == "invalid":
        return GGUFTensorIndexReport("invalid", (), ("invalid_catalog",))
    rows = tuple(_inspect(x, root) for x in declaration.models if x.format == "gguf")
    if not rows:
        return GGUFTensorIndexReport("no_gguf", ())
    return GGUFTensorIndexReport("inspected" if all(x.status == "index_checked" for x in rows) else "partial", rows)
