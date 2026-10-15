"""Offline Nsight Systems kernel-name report; no CUDA is executed here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from .dependencies import PROJECT_ROOT


def classify_kernel(name: str) -> str | None:
    n = name.lower()
    # Names verified against pinned llama.cpp 0c1e5709, not guessed from
    # user-facing Flash ON/OFF settings. Check helpers before broad matching.
    for marker, family in (
        ("flash_attn_ext_vec", "FLASH_VEC"),
        ("flash_attn_ext_f16", "FLASH_MMA_F16"),
        ("flash_attn_ext_tile", "FLASH_TILE"),
        ("flash_attn_stream_k_fixup", "FLASH_FIXUP"),
        ("flash_attn_combine_results", "FLASH_COMBINE"),
    ):
        if marker in n:
            return family
    return "FLASH_OTHER" if "flash_attn" in n or "fattn" in n else None


def classify_observed_operator(name: str) -> str | None:
    """Kernel-level evidence, not proof of a specific model graph node.

    On pinned llama.cpp, Flash OFF builds K@Q, softmax, V@attention. CUTLASS
    GEMM names can ALSO represent ordinary projection matrices.
    """
    n = name.lower()
    if "soft_max_f32" in n and "soft_max_back_f32" not in n:
        return "SOFTMAX_F32"
    if "cutlass::kernel2<" in n:
        return "CUTLASS_GEMM_UNATTRIBUTED"
    return None


def inspect_export(
    file: Path, root: Path = PROJECT_ROOT, *,
    expected_flash: str | None = None,
) -> dict:
    if expected_flash not in (None, "on", "off"):
        raise ValueError("Expected Flash mode must be on/off or omitted")
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
                "operator_family": classify_observed_operator(name),
            })
    matches = [r for r in result if r["attention_family"]]
    primary = {"FLASH_VEC", "FLASH_MMA_F16", "FLASH_TILE"}
    main = [r for r in matches if r["attention_family"] in primary]
    summary = []
    for family in sorted({r["attention_family"] for r in matches}):
        entries = [r for r in matches if r["attention_family"] == family]
        summary.append({
            "attention_family": family,
            "role": "main" if family in primary else "auxiliary_or_other",
            "count": sum(r["count"] for r in entries),
            "kernel_gpu_ms": round(sum(r["kernel_gpu_ms"] for r in entries), 4),
        })
    nonflash_ops = [r for r in result if r["operator_family"] is not None]
    operator_summary = []
    for family in sorted({r["operator_family"] for r in nonflash_ops}):
        entries = [r for r in nonflash_ops if r["operator_family"] == family]
        operator_summary.append({
            "operator_family": family,
            "count": sum(r["count"] for r in entries),
            "kernel_gpu_ms": round(sum(r["kernel_gpu_ms"] for r in entries), 4),
        })
    has_softmax = any(r["operator_family"] == "SOFTMAX_F32"
                      for r in nonflash_ops)
    has_cutlass = any(r["operator_family"] == "CUTLASS_GEMM_UNATTRIBUTED"
                      for r in nonflash_ops)
    # Source-informed checks that remain weaker than a graph-to-kernel join.
    expected_match = (
        None if expected_flash is None else
        bool(main) if expected_flash == "on" else
        not bool(matches) and has_softmax and has_cutlass
    )
    return {
        "status": "cuda_kernel_names_observed" if result else "no_kernel_records",
        "source_revision_for_name_mapping":
            "0c1e57098bba43ac29e6e3b677cdceebdd22334f",
        "kernel_records_total": sum(r["count"] for r in result),
        "attention_kernel_records": matches,
        "attention_family_summary": summary,
        "operator_kernel_summary": operator_summary,
        "nonflash_softmax_kernel_observed": has_softmax,
        "cutlass_gemm_kernels_observed_not_attributed": has_cutlass,
        "expected_flash": expected_flash,
        "expected_flash_consistent_with_kernel_names": expected_match,
        "nonflash_graph_kernel_attribution_verified": False,
        "main_attention_families_observed": sorted({
            r["attention_family"] for r in main}),
        "kernel_name_sample": result[:80],
        "native_kernel_names_verified": bool(main),
        "prefill_decode_attribution_verified": False,
        "production_qualified": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline Nsight CUDA kernel report")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--expected-flash", choices=("on", "off"),
                   help="Optional external experiment label; fail closed when "
                        "its kernel evidence is inconsistent")
    args = p.parse_args(argv)
    try:
        output = inspect_export(Path(args.sqlite),
                                expected_flash=args.expected_flash)
    except (ValueError, OSError, sqlite3.Error) as error:
        output = {"status": "dispatch_export_unverified",
                  "error": type(error).__name__ + ": " + str(error)[:200],
                  "native_kernel_names_verified": False,
                  "production_qualified": False}
    print(json.dumps(output, sort_keys=True, indent=2))
    if args.expected_flash is not None:
        return 0 if output.get("expected_flash_consistent_with_kernel_names") is True else 2
    return 0 if output.get("native_kernel_names_verified") else 2


if __name__ == "__main__":
    raise SystemExit(main())
