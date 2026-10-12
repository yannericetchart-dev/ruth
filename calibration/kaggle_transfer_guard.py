"""Only check disk capacity for the missing shards of a Kaggle GGUF.

VRAM filtering occurs in baseline_discovery from the SUM of all shards'
catalog file sizes, not here. There is no download allowance or budget.
"""

import os
from pathlib import Path
import shutil

MIB = 1048576
DISK_RESERVE_MIB = 1024


def _existing_ancestor(path: Path) -> Path:
    current = Path(path).expanduser()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError("Impossibile determinare il filesystem Kaggle.")
        current = parent
    return current


def assess_kaggle_transfer(
    candidate: dict, destination: Path, *,
    cache_root: Path | None = None,
) -> dict:
    """Estimate free space for KaggleHub cache plus local shard copies."""
    destination = Path(destination).expanduser()
    names = candidate.get("shard_filenames") or [candidate["filename"]]
    missing = [
        i for i, name in enumerate(names)
        if not (destination / name).is_file()
    ]
    if not missing:
        return {"allowed": True, "missing_shards": 0, "catalog_bytes": 0}

    sizes = candidate.get("reported_shard_sizes") or []
    if len(sizes) == len(names) and all(
        isinstance(sizes[i], dict)
        and isinstance(sizes[i].get("bytes"), int)
        and not isinstance(sizes[i]["bytes"], bool)
        and sizes[i]["bytes"] > 0 for i in missing
    ):
        bytes_needed = sum(sizes[i]["bytes"] for i in missing)
    elif len(missing) == len(names) and isinstance(
        candidate.get("reported_download_bytes"), int
    ) and candidate["reported_download_bytes"] > 0:
        bytes_needed = candidate["reported_download_bytes"]
    else:
        return {
            "allowed": False, "status": "blocked_kaggle_unknown_disk_size",
            "reason": "Dimensione degli shard Kaggle mancanti non verificabile.",
        }

    cache = cache_root or Path(os.environ.get(
        "KAGGLEHUB_CACHE", str(Path.home() / ".cache" / "kagglehub")
    ))
    target_mount = _existing_ancestor(destination)
    cache_mount = _existing_ancestor(Path(cache))
    same_disk = target_mount.stat().st_dev == cache_mount.stat().st_dev
    reserve = DISK_RESERVE_MIB * MIB
    # Estimate one cached copy and one local copy, +5% on catalog sizes.
    estimated = (bytes_needed * 105 + 99) // 100
    free_target = shutil.disk_usage(target_mount).free
    free_cache = shutil.disk_usage(cache_mount).free
    need_target = estimated * (2 if same_disk else 1) + reserve
    if free_target < need_target or (
        not same_disk and free_cache < estimated + reserve
    ):
        return {
            "allowed": False,
            "status": "blocked_kaggle_insufficient_disk",
            "reason": (
                "Spazio insufficiente per cache KaggleHub e copie locali "
                f"({round(bytes_needed/MIB, 1)} MiB da catalogo)."
            ),
        }
    return {
        "allowed": True, "missing_shards": len(missing),
        "catalog_bytes": bytes_needed,
        "same_disk_for_cache_and_copy": same_disk,
    }
