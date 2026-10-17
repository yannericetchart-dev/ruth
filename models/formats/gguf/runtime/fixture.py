"""Synthetic end-to-end smoke check for existing GGUF metadata readers.

Disposable local fixture files contain bounded zero placeholders, never models.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct

from models.catalog import catalog
from .reader import inspect_headers
from .metadata import inspect_all_metadata
from .tensor_index import inspect_all_tensor_indexes
from .blocks import inspect_all_block_layouts
from .kblocks import inspect_all_kblock_layouts


@dataclass(frozen=True)
class GGUFIndirectFixtureReport:
    status: str
    fixtures: int = 0
    stages: tuple[str, ...] = ()
    reason: str = ""


def _synthetic_gguf(*, version: int, order: str, kind: int, width: int, size: int) -> bytes:
    key, value, name = b"general.architecture", b"llama", b"fixture.tensor"
    metadata = (struct.pack(order + "Q", len(key)) + key +
                struct.pack(order + "I", 8) + struct.pack(order + "Q", len(value)) + value)
    descriptor = (struct.pack(order + "Q", len(name)) + name +
                  struct.pack(order + "I", 1) + struct.pack(order + "Q", width) +
                  struct.pack(order + "IQ", kind, 0))
    header = b"GGUF" + struct.pack(order + "IQQ", version, 1, 1)
    structure = header + metadata + descriptor
    return structure + b"\x00" * (-len(structure) % 32) + b"\x00" * size


def _create_fixture_catalog(root: Path) -> None:
    """Private helper; use ONLY with a disposable, empty project root."""
    folder = root / ".local" / "models"
    folder.mkdir(parents=True, exist_ok=False)
    specs = (
        ("scalar", 2, "<", 0, 4, 16),
        ("q4", 3, ">", 2, 32, 18),
        ("kquant", 3, "<", 12, 256, 144),
    )
    entries = []
    for ident, version, order, kind, width, size in specs:
        filename = ident + ".gguf"
        (folder / filename).write_bytes(
            _synthetic_gguf(version=version, order=order, kind=kind,
                            width=width, size=size))
        entries.append({"id": ident, "format": "gguf", "path": filename})
    (folder / "catalog.json").write_text(json.dumps({"models": entries}), encoding="utf-8")


def run_fixture_pipeline() -> GGUFIndirectFixtureReport:
    """Verify only temporary declared fixture files; do not read tensor payload.

    INDEX and BLOCKS intentionally report unverified quantized sizes.
    """
    try:
        with TemporaryDirectory(prefix="ruth-gguf-fixture-") as temp:
            root = Path(temp)
            _create_fixture_catalog(root)
            cat = catalog(project_root=root)
            headers = inspect_headers(project_root=root)
            metadata = inspect_all_metadata(project_root=root)
            index = inspect_all_tensor_indexes(project_root=root)
            blocks = inspect_all_block_layouts(project_root=root)
            kblocks = inspect_all_kblock_layouts(project_root=root)
            stages = (
                "CAT=" + cat.status, "GGUF=" + headers.status,
                "META=" + metadata.status, "INDEX=" + index.status,
                "BLOCKS=" + blocks.status, "KBLOCKS=" + kblocks.status
            )
            ok = (
                cat.status == "ready" and cat.count == 3 and
                headers.status == "inspected" and len(headers.headers) == 3 and
                all(x.status == "header_read" for x in headers.headers) and
                metadata.status == "inspected" and len(metadata.models) == 3 and
                all(x.status == "metadata_read" for x in metadata.models) and
                index.status == "partial" and
                tuple(x.status for x in index.models) ==
                ("index_checked", "size_unverified", "size_unverified") and
                blocks.status == "partial" and
                tuple(x.status for x in blocks.models) ==
                ("layout_checked", "layout_checked", "size_unverified") and
                kblocks.status == "inspected" and len(kblocks.models) == 3 and
                all(x.status == "kblocks_checked" and x.size_unverified == 0
                    for x in kblocks.models)
            )
            return GGUFIndirectFixtureReport(
                "passed" if ok else "failed", 3, stages,
                "" if ok else "unexpected_report"
            )
    except (OSError, ValueError):
        return GGUFIndirectFixtureReport("unavailable", reason="fixture_unavailable")
