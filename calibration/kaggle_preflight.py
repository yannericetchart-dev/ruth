"""Kaggle GGUF transfer preflight: metadata only, no weight downloads.

A listed file size is NOT tensor payload or a CUDA feasibility result.
This intentionally cannot skip a higher-quality candidate silently.
"""

import argparse
from pathlib import Path
import shutil

from hardware import GPUManager
from model import ModelManager
from .kaggle_source import inventory_kaggle
from .profile_pipeline import atomic_json


def inspect_transfer_plan(model_id: str = "qwen-2.5-7b-instruct",
                          *, gpu_status: dict | None = None,
                          disk_path: Path | None = None,
                          selected_handle: str | None = None) -> dict:
    spec = ModelManager.source_spec(model_id, "kaggle")
    if gpu_status is None:
        gpu_status = GPUManager.calibration_status()
    if not gpu_status.get("ready"):
        raise RuntimeError("GPU non pronta per la pianificazione Kaggle.")
    handle = selected_handle or spec["kaggle_handle"]
    models = inventory_kaggle(
        spec["repository"], spec["gguf_prefix"], selected_handle=handle
    )
    manager = ModelManager()
    dest = disk_path or manager.model_directory(
        model_id, "kaggle", spec["repository"]
    )
    existing = dest
    while not existing.exists():
        if existing == existing.parent:
            raise RuntimeError("Nessun volume disco accessibile.")
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    vram_bytes = int(gpu_status["memory"]["total_mib"] * 1048576)
    rows = []
    for candidate in models:
        bytes_reported = candidate.get("reported_download_bytes")
        all_complete = candidate.get("split_complete") is not False
        rows.append({
            "quantization": candidate["quantization"],
            "first_shard": candidate["filename"],
            "shard_count": candidate.get("shard_count", 1),
            "shards_complete": all_complete,
            "quality_priority": candidate["quality_priority"],
            "reported_download_bytes": bytes_reported,
            "reported_download_mib": (
                round(bytes_reported / 1048576, 2)
                if bytes_reported is not None else None
            ),
            "reported_size_estimated": candidate.get(
                "reported_download_size_estimated"
            ),
            "reported_file_total_above_vram": (
                bytes_reported > vram_bytes
                if bytes_reported is not None else None
            ),
            "fits_available_disk_as_single_copy": (
                bytes_reported <= free
                if bytes_reported is not None else None
            ),
            "cpu_gpu_load_feasibility": "not_determined_from_file_sizes",
            "preflight_download_performed": False,
        })
    return {
        "model_id": model_id, "source": "kaggle", "handle": handle,
        "gpu_total_mib": round(vram_bytes / 1048576, 1),
        "available_disk_mib": round(free / 1048576, 1),
        "candidates": rows, "weights_downloaded": False,
        "native_gpu_probe_executed": False,
        "warning": (
            "Kaggle: dimensione GGUF = somma delle dimensioni catalogo "
            "dei suoi shard. Se supera la VRAM totale, la baseline "
            "lo salta prima del download. Questo e' un filtro "
            "pratico sulle dimensioni dei file, non una prova "
            "matematica sui byte effettivi dei tensori: "
            "dimensioni catalogo arrotondate possono dare falsi "
            "positivi vicino alla soglia. I candidati non esclusi "
            "richiedono comunque un test nativo full-CUDA."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preflight dimensioni file Kaggle (solo metadati)."
    )
    parser.add_argument("--model", default="qwen-2.5-7b-instruct",
                        choices=["qwen-2.5-7b-instruct"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    plan = inspect_transfer_plan(args.model)
    dest = (args.output or
            ModelManager().root / "calibration_runs" / "kaggle_preflight.json")
    atomic_json(dest, plan)
    print(f"KAGGLE | {plan['model_id']}: "
          f"VRAM={plan['gpu_total_mib']} MiB, "
          f"disco libero={plan['available_disk_mib']} MiB")
    for item in plan["candidates"]:
        size = (f"{item['reported_download_mib']} MiB"
                if item["reported_download_mib"] is not None else "SCONOSCIUTA")
        if item["reported_file_total_above_vram"] is True:
            risk = "GGUF totale > VRAM: salta il download"
        elif item["reported_file_total_above_vram"] is None:
            risk = "dimensione GGUF sconosciuta: blocca baseline"
        else:
            risk = "GGUF totale <= VRAM: prova CUDA necessaria"
        if not item["shards_complete"]:
            risk = "SHARD INCOMPLETI"
        print(f"  {item['quantization']:10} | "
              f"{item['shard_count']} shard | "
              f"dimensione catalogo={size} | {risk}")
    print(
        f"Report: {dest} | Nessun download o caricamento GPU."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
