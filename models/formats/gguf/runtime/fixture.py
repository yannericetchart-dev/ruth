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
from .structural_summary import inspect_structural_summaries


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



def _synthetic_multitensor(*, specs: tuple[tuple[str, int, int, int], ...] | None = None,
                           alignment: int = 32, data_size: int = 208,
                           version: int = 3, order: str = "<") -> bytes:
    """Build multi-descriptor GGUF with bounded zero placeholders, never weights.

    Each spec is (unique name, first dimension, GGML type, relative byte offset).
    """
    if specs is None:
        specs = (
            ("multi.scalar", 4, 0, 0),      # 4 * 4 = 16 bytes
            ("multi.q4", 32, 2, 32),       # Q4_0 = 18 bytes
            ("multi.kquant", 256, 12, 64), # Q4_K = 144 bytes
        )
    if not 1 <= len(specs) <= 16 or not 8 <= alignment <= 1024 or alignment % 8:
        raise ValueError("invalid_synthetic_fixture")
    if data_size < 0 or data_size > 4096:
        raise ValueError("invalid_synthetic_fixture")
    arch, value, key = b"general.architecture", b"llama", b"general.alignment"
    metadata = (
        struct.pack(order + "Q", len(arch)) + arch + struct.pack(order + "I", 8) +
        struct.pack(order + "Q", len(value)) + value +
        struct.pack(order + "Q", len(key)) + key + struct.pack(order + "II", 4, alignment)
    )
    descriptors = b""
    for name, width, kind, offset in specs:
        encoded = name.encode("ascii")
        descriptors += (
            struct.pack(order + "Q", len(encoded)) + encoded +
            struct.pack(order + "IQIQ", 1, width, kind, offset)
        )
    structure = (b"GGUF" + struct.pack(order + "IQQ", version, len(specs), 2) +
                 metadata + descriptors)
    return (structure + b"\x00" * (-len(structure) % alignment) +
            b"\x00" * data_size)


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
        if ident == "kquant":
            (folder / "multi.gguf").write_bytes(_synthetic_multitensor())
            entries.append({"id": "multi", "format": "gguf", "path": "multi.gguf"})
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
            summary = inspect_structural_summaries(project_root=root)
            stages = (
                "CAT=" + cat.status, "GGUF=" + headers.status,
                "META=" + metadata.status, "INDEX=" + index.status,
                "BLOCKS=" + blocks.status, "KBLOCKS=" + kblocks.status,
                "SUMMARY=" + summary.status
            )
            ok = (
                cat.status == "ready" and cat.count == 4 and
                headers.status == "inspected" and len(headers.headers) == 4 and
                all(x.status == "header_read" for x in headers.headers) and
                metadata.status == "inspected" and len(metadata.models) == 4 and
                all(x.status == "metadata_read" for x in metadata.models) and
                index.status == "partial" and
                tuple(x.status for x in index.models) ==
                ("index_checked", "size_unverified", "size_unverified", "size_unverified") and
                blocks.status == "partial" and
                tuple(x.status for x in blocks.models) ==
                ("layout_checked", "layout_checked", "size_unverified", "size_unverified") and
                kblocks.status == "inspected" and len(kblocks.models) == 4 and
                all(x.status == "kblocks_checked" and x.size_unverified == 0
                    for x in kblocks.models) and
                summary.status == "structural_checked" and
                tuple((x.model_id, x.status, x.tensor_count, x.size_unverified, x.reason)
                      for x in summary.models) == (
                          ("scalar", "structural_checked", 1, 0, ""),
                          ("q4", "structural_checked", 1, 0, ""),
                          ("multi", "structural_checked", 3, 0, ""),
                          ("kquant", "structural_checked", 1, 0, ""),
                      )
            )
            return GGUFIndirectFixtureReport(
                "passed" if ok else "failed", 4, stages,
                "" if ok else "unexpected_report"
            )
    except (OSError, ValueError):
        return GGUFIndirectFixtureReport("unavailable", reason="fixture_unavailable")
