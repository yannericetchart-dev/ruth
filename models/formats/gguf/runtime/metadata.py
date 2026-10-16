"""Bounded, catalog-bound GGUF metadata inspection (no tensors or inference).

All failures are portable codes. Metadata values are transient and never
persisted; only three approved technical fields are exposed.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from models.catalog import ModelRecord, catalog
from .reader import _root, _safe_path

_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_FIELDS = 16384
_MAX_STRING_BYTES = 8 * 1024 * 1024
_MAX_ARRAY_ITEMS = 500000
_MAX_TOTAL_ARRAY_ITEMS = 1000000
_MAX_DEPTH = 6
_KEY = re.compile(r"^[a-z_][a-z_0-9]*(?:\.[a-z_][a-z_0-9]*)+$")
_ARCH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4,
          7: 1, 10: 8, 11: 8, 12: 8}


@dataclass(frozen=True)
class GGUFMetadata:
    """Selected unverified declarations, not a model compatibility report."""

    model_id: str
    status: str
    architecture: str | None = None
    file_type: int | None = None
    context_length: int | None = None


@dataclass(frozen=True)
class GGUFMetadataReport:
    status: str  # not_configured / empty / no_gguf / inspected / partial / invalid
    models: tuple[GGUFMetadata, ...]
    errors: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return len(self.models)


class _ParseError(Exception):
    def __init__(self, code: str):
        self.code = code


class _Cursor:
    def __init__(self, file):
        self.file = file
        self.size = os.fstat(file.fileno()).st_size
        self.used = 0
        self.total_items = 0

    def _check(self, length: int):
        if length < 0 or self.used + length > _MAX_METADATA_BYTES:
            raise _ParseError("limit_exceeded")
        if self.file.tell() + length > self.size:
            raise _ParseError("truncated_metadata")
        self.used += length

    def read(self, length: int) -> bytes:
        self._check(length)
        data = self.file.read(length)
        if len(data) != length:
            raise _ParseError("truncated_metadata")
        return data

    def skip(self, length: int):
        self._check(length)
        self.file.seek(length, os.SEEK_CUR)

    def number(self, width: int, order: str) -> int:
        return int.from_bytes(self.read(width), order)

    def string(self, order: str, *, capture: bool = False) -> str | None:
        size = self.number(8, order)
        if size > _MAX_STRING_BYTES:
            raise _ParseError("limit_exceeded")
        if not capture:
            self.skip(size)
            return None
        try:
            return self.read(size).decode("utf-8")
        except UnicodeDecodeError:
            raise _ParseError("invalid_metadata") from None

    def value(self, kind: int, order: str, *, capture: bool = False, depth: int = 0):
        if kind in _SIZES:
            width = _SIZES[kind]
            if capture:
                return self.number(width, order)
            self.skip(width)
            return None
        if kind == 8:
            return self.string(order, capture=capture)
        if kind != 9:
            raise _ParseError("unsupported_type")
        if depth >= _MAX_DEPTH:
            raise _ParseError("limit_exceeded")
        item_type = self.number(4, order)
        count = self.number(8, order)
        if count > _MAX_ARRAY_ITEMS or self.total_items + count > _MAX_TOTAL_ARRAY_ITEMS:
            raise _ParseError("limit_exceeded")
        self.total_items += count
        if item_type in _SIZES:
            self.skip(count * _SIZES[item_type])
        else:
            for _ in range(count):
                self.value(item_type, order, depth=depth + 1)
        return None


def _parse(file) -> tuple[str | None, int | None, int | None]:
    cursor = _Cursor(file)
    header = cursor.read(24)
    if header[:4] != b"GGUF":
        raise _ParseError("invalid_magic")
    little = int.from_bytes(header[4:8], "little")
    big = int.from_bytes(header[4:8], "big")
    if little in (2, 3):
        order = "little"
    elif big == 3:
        order = "big"
    else:
        raise _ParseError("unsupported_version")
    declared = int.from_bytes(header[16:24], order)
    if declared > _MAX_FIELDS:
        raise _ParseError("limit_exceeded")
    architecture = None
    file_type = None
    contexts: dict[str, int] = {}
    seen: set[str] = set()
    for _ in range(declared):
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
        target = key in ("general.architecture", "general.file_type") or key.endswith(".context_length")
        if target and key in seen:
            raise _ParseError("invalid_metadata")
        if target:
            seen.add(key)
        if key == "general.architecture":
            if kind != 8:
                raise _ParseError("invalid_metadata")
            architecture = cursor.value(kind, order, capture=True)
            if not _ARCH.fullmatch(architecture):
                raise _ParseError("invalid_metadata")
        elif key == "general.file_type":
            if kind != 4:
                raise _ParseError("invalid_metadata")
            file_type = cursor.value(kind, order, capture=True)
        elif key.endswith(".context_length"):
            if kind not in (4, 10):
                raise _ParseError("invalid_metadata")
            if len(contexts) >= 64:
                raise _ParseError("limit_exceeded")
            contexts[key] = cursor.value(kind, order, capture=True)
        else:
            cursor.value(kind, order)
    context = contexts.get(architecture + ".context_length") if architecture else None
    return architecture, file_type, context


def _inspect(record: ModelRecord, root: Path) -> GGUFMetadata:
    if record.format != "gguf":
        return GGUFMetadata(record.id, "not_gguf")
    if record.status == "missing":
        return GGUFMetadata(record.id, "missing")
    path = _safe_path(root, record.reference)
    if path is None:
        return GGUFMetadata(record.id, "unsafe_reference")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            architecture, file_type, context = _parse(stream)
        return GGUFMetadata(record.id, "metadata_read", architecture, file_type, context)
    except _ParseError as error:
        return GGUFMetadata(record.id, error.code)
    except (OSError, ValueError, OverflowError):
        return GGUFMetadata(record.id, "unreadable")


def inspect_metadata(model_id: str, *, project_root: Path | None = None) -> GGUFMetadata:
    """Read only an id explicitly declared in CAT; never accept a path."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status == "invalid":
        return GGUFMetadata(model_id, "invalid_catalog")
    match = next((row for row in declared.models if row.id == model_id), None)
    return GGUFMetadata(model_id, "not_declared") if match is None else _inspect(match, root)


def inspect_all_metadata(*, project_root: Path | None = None) -> GGUFMetadataReport:
    """Inspect metadata for each declared GGUF record, with no discovery."""
    root = _root(project_root)
    declared = catalog(project_root=root)
    if declared.status in ("not_configured", "empty"):
        return GGUFMetadataReport(declared.status, ())
    if declared.status == "invalid":
        return GGUFMetadataReport("invalid", (), ("invalid_catalog",))
    gguf = (row for row in declared.models if row.format == "gguf")
    rows = tuple(_inspect(row, root) for row in gguf)
    if not rows:
        return GGUFMetadataReport("no_gguf", ())
    state = "inspected" if all(row.status == "metadata_read" for row in rows) else "partial"
    return GGUFMetadataReport(state, rows)
