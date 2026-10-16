"""Explicit local JSON model manifest, strictly confined to .local/models."""
from __future__ import annotations

import json
import re
from pathlib import Path
from .contracts import ModelCatalog, ModelRecord

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_FORMAT = re.compile(r"^[a-z][a-z0-9_+.-]{0,31}$")
_MAX_BYTES = 1024 * 1024
_MAX_RECORDS = 200
_MAX_REFERENCE = 240


def _project_root(project_root: Path | None) -> Path:
    return Path(__file__).resolve().parents[3] if project_root is None else Path(project_root)


def _is_safe_file(base: Path, relative: Path) -> bool:
    """Avoid symlinks in every component, including the root of local data."""
    current = base
    if base.is_symlink():
        return False
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return False
    return True


def _valid_reference(value: object) -> Path | None:
    """Allow only explicitly listed, POSIX-style, relative model files."""
    if not isinstance(value, str) or not value or len(value) > _MAX_REFERENCE:
        return None
    if "\\" in value or "\x00" in value or value.startswith("/"):
        return None
    parts = value.split("/")
    if any(part in ("", ".", "..") or part.startswith(".") for part in parts):
        return None
    if any(":" in part or any(ord(ch) < 32 or ord(ch) == 127 for ch in part) for part in parts):
        return None
    return Path(*parts)


def catalog(*, project_root: Path | None = None) -> ModelCatalog:
    """Inspect only .local/models/catalog.json and its explicitly listed paths.

    No download, recursive search, model import or binary format inspection.
    All error messages are portable codes and never echo local data.
    """
    root = _project_root(project_root)
    base = root / ".local"
    manifest = base / "models" / "catalog.json"
    if not _is_safe_file(root, Path(".local/models/catalog.json")):
        return ModelCatalog("invalid", (), ("unsafe_manifest_path",))
    try:
        if not manifest.exists():
            return ModelCatalog("not_configured", (), ())
        if not manifest.is_file() or manifest.stat().st_size > _MAX_BYTES:
            return ModelCatalog("invalid", (), ("invalid_manifest",))
        with manifest.open("rb") as file:
            raw = file.read(_MAX_BYTES + 1)
        if len(raw) > _MAX_BYTES:
            return ModelCatalog("invalid", (), ("manifest_too_large",))
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        return ModelCatalog("invalid", (), ("unreadable_manifest",))
    if not isinstance(payload, dict) or set(payload) != {"models"} or not isinstance(payload["models"], list):
        return ModelCatalog("invalid", (), ("invalid_schema",))
    entries = payload["models"]
    if len(entries) > _MAX_RECORDS:
        return ModelCatalog("invalid", (), ("too_many_models",))
    if not entries:
        return ModelCatalog("empty", (), ())

    result: list[ModelRecord] = []
    seen: set[str] = set()
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"id", "format", "path"}:
            return ModelCatalog("invalid", (), ("invalid_entry",))
        identifier, fmt = row["id"], row["format"]
        rel = _valid_reference(row["path"])
        if not isinstance(identifier, str) or not _ID.fullmatch(identifier):
            return ModelCatalog("invalid", (), ("invalid_id",))
        if identifier in seen:
            return ModelCatalog("invalid", (), ("duplicate_id",))
        if not isinstance(fmt, str) or not _FORMAT.fullmatch(fmt):
            return ModelCatalog("invalid", (), ("invalid_format",))
        if rel is None:
            return ModelCatalog("invalid", (), ("unsafe_reference",))
        if not _is_safe_file(base, Path("models") / rel):
            return ModelCatalog("invalid", (), ("unsafe_reference",))
        seen.add(identifier)
        file_path = base / "models" / rel
        try:
            present = file_path.is_file()
        except (OSError, ValueError):
            present = False
        result.append(ModelRecord(identifier, fmt, rel.as_posix(),
                                  "present_unverified" if present else "missing"))
    state = "ready" if all(model.status == "present_unverified" for model in result) else "partial"
    return ModelCatalog(state, tuple(result), ())


def resolve(reference: str, *, project_root: Path | None = None) -> ModelRecord | None:
    """Resolve an explicitly declared id; never attempt discovery or download."""
    if not isinstance(reference, str) or not _ID.fullmatch(reference):
        raise ValueError("invalid_reference")
    report = catalog(project_root=project_root)
    if report.status == "invalid":
        raise ValueError("invalid_catalog")
    return next((model for model in report.models if model.id == reference), None)
