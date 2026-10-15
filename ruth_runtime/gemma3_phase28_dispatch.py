"""Offline Nsight Systems kernel-name report; no CUDA is executed here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT


def classify_kernel(name: str) -> str | None:
    n = name.lower()
    for marker, family in (
        ("flash_attn_ext_vec", "FLASH_VEC"),
        ("flash_attn_ext_mma_f16", "FLASH_MMA_F16"),
        ("flash_attn_ext_tile", "FLASH_TILE"),
    ):
        if marker in n:
            return family
    return "FLASH_OTHER" if "flash_attn" in n or "fattn" in n else None


def inspect_export(file: Path, root: Path = PROJECT_ROOT) -> dict:
    base = (root / ".ruth-build/phase28").resolve(strict=True)
    file = (root / file) if not file.is_absolute() else file
    if (file.is_symlink() or not file.is_file()
        or base not in file.resolve().parents or file.suffix != ".sqlite"
        or not 0 < file.stat().st_size <= 1_000_000_000):
        raise ValueError("Expected bounded local Phase28 SQLite export")
    with sqlite3.connect(file.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"} <= names:
            raise ValueError("No Nsight CUDA kernel records")
        cols = {row[1] for row in conn.execute(
            "PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        col = "demangledName" if "demangledName" in cols else (
            "shortName" if "shortName" in cols else "")
        if not col or not {"start", "end"} <= cols:
            raise ValueError("Unrecognized Nsight kernel schema")
        statement = (
            "SELECT s.value, COUNT(*), SUM(k.end-k.start) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON s.id=k." + col +
            " WHERE k.end>=k.start GROUP BY s.value"
        )
        result = []
        for name, count, nanos in conn.execute(statement):
            if not isinstance(name, str) or count <= 0 or nanos < 0:
                raise ValueError("Malformed kernel activity record")
            result.append({
                "name": name[:500], "count": int(count),
                "kernel_gpu_ms": round(nanos / 1e6, 4),
                "attention_family": classify_kernel(name),
            })
    matches = [r for r in result if r["attention_family"]]
    return {
        "status": "cuda_kernel_names_observed" if result else "no_kernel_records",
        "kernel_records_total": sum(r["count"] for r in result),
        "attention_kernel_records": matches,
        "kernel_name_sample": result[:80],
        "native_kernel_names_verified": bool(matches),
        "prefill_decode_attribution_verified": False,
        "production_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline Nsight CUDA kernel report")
    p.add_argument("--sqlite", required=True)
    args = p.parse_args(argv)
    try:
        output = inspect_export(Path(args.sqlite))
    except (ValueError, OSError, sqlite3.Error) as error:
        output = {"status": "dispatch_export_unverified",
                  "error": type(error).__name__ + ": " + str(error)[:200],
                  "native_kernel_names_verified": False,
                  "production_qualified": False}
    print(json.dumps(output, sort_keys=True, indent=2))
    return 0 if output.get("native_kernel_names_verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
