"""Bounded 24-byte GGUF header inspection of catalog-declared local models.

Version 2 uses little endian; version 3 also permits big endian.
Never load metadata, tensors, model data, or network resources.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct

from models.catalog import ModelRecord, catalog
from .contracts import GGUFHeader, GGUFReport

_HEADER_SIZE = 24
_SUPPORTED = (2, 3)


def _root(root: Path | None) -> Path:
    return Path(__file__).resolve().parents[4] if root is None else Path(root)


def _safe_path(root: Path, reference: str) -> Path | None:
    """Recheck the catalog-relative reference and every symlink before opening."""
    parts = reference.split("/") if isinstance(reference, str) else []
    if not parts or any(not part or part in (".", "..") or part.startswith(".") or
                        "\\" in part or ":" in part for part in parts):
        return None
    current = root
    for part in (".local", "models", *parts):
        current = current / part
        try:
            if current.is_symlink():
                return None
        except (OSError, ValueError):
            return None
    return current


def _inspect(record: ModelRecord, root: Path) -> GGUFHeader:
    if record.format != "gguf":
        return GGUFHeader(record.id, "not_gguf")
    if record.status == "missing":
        return GGUFHeader(record.id, "missing")
    target = _safe_path(root, record.reference)
    if target is None:
        return GGUFHeader(record.id, "unsafe_reference")
    try:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(target, flags)
        with os.fdopen(fd, "rb") as file:
            raw = file.read(_HEADER_SIZE)
    except (OSError, ValueError):
        return GGUFHeader(record.id, "unreadable")
    if len(raw) < _HEADER_SIZE:
        return GGUFHeader(record.id, "truncated_header")
    if raw[:4] != b"GGUF":
        return GGUFHeader(record.id, "invalid_magic")
    little_version = int.from_bytes(raw[4:8], "little")
    big_version = int.from_bytes(raw[4:8], "big")
    if little_version in _SUPPORTED:
        fmt, order, version = "<IQQ", "little", little_version
    elif big_version == 3:
        fmt, order, version = ">IQQ", "big", big_version
    else:
        return GGUFHeader(record.id, "unsupported_version")
    version, tensors, metadata = struct.unpack(fmt, raw[4:])
    return GGUFHeader(record.id, "header_read", version, tensors, metadata, order)


def inspect_headers(*, project_root: Path | None = None) -> GGUFReport:
    """Inspect only GGUF records already declared and screened by CAT-001."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status in ("not_configured", "empty"):
        return GGUFReport(declared.status, ())
    if declared.status == "invalid":
        return GGUFReport("invalid", (), ("invalid_catalog",))
    ggufs = tuple(row for row in declared.models if row.format == "gguf")
    if not ggufs:
        return GGUFReport("no_gguf", ())
    rows = tuple(_inspect(record, root) for record in ggufs)
    status = "inspected" if all(row.status == "header_read" for row in rows) else "partial"
    return GGUFReport(status, rows)


def inspect_header(model_id: str, *, project_root: Path | None = None) -> GGUFHeader:
    """Inspect one declared model identifier; arbitrary paths are not accepted."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status == "invalid":
        return GGUFHeader(model_id, "invalid_catalog")
    match = next((row for row in declared.models if row.id == model_id), None)
    if match is None:
        return GGUFHeader(model_id, "not_declared")
    return _inspect(match, root)
