"""Conservative, catalogue-based approval before Kaggle GGUF transfers.

Never infer tensor payload or CUDA feasibility from catalogue size.
All shard transfer estimates are an approval gate, NOT a byte-accurate
network quota (Kaggle may round sizes or reuse its cache).
"""

import os
from pathlib import Path
import shutil


MIB = 1024 * 1024
DEFAULT_MIN_FREE_MIB = 8192


def _existing_ancestor(path: Path) -> Path:
    current = Path(path).expanduser()
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise RuntimeError("Impossibile determinare il filesystem per Kaggle.")
        current = parent
    return current


def assess_kaggle_transfer(
    candidate: dict, destination: Path, *,
    candidate_budget_bytes: int | None,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
    cache_root: Path | None = None,
) -> dict:
    """Approve a group of *missing* shards within this candidate's budget.
    
    The budget is reset for each quantization. It is not shared with earlier
    or later candidates. This function itself never downloads any bytes.

    The default cache root follows KaggleHub's usual user cache. An explicit
    KAGGLEHUB_CACHE environment variable takes precedence. The disk estimate
    conservatively assumes an uncompressed cached copy plus a destination
    copy, with 5% allowance for catalogue rounding and a free-space reserve.
    """
    if (candidate_budget_bytes is not None and
        (not isinstance(candidate_budget_bytes, int)
         or isinstance(candidate_budget_bytes, bool)
         or candidate_budget_bytes < 0)):
        raise ValueError("Budget di trasferimento Kaggle non valido.")
    if (not isinstance(min_free_mib, int)
        or isinstance(min_free_mib, bool) or min_free_mib < 0):
        raise ValueError("Riserva minima disco Kaggle non valida.")
    target_dir = Path(destination).expanduser()
    shard_names = candidate.get("shard_filenames") or [candidate["filename"]]
    shard_sizes = candidate.get("reported_shard_sizes") or []
    missing = [
        (index, name) for index, name in enumerate(shard_names)
        if not (target_dir / name).is_file()
    ]
    if not missing:
        return {
            "allowed": True, "catalog_transfer_bytes": 0,
            "missing_shards": 0, "reason": "already_local",
        }
    if candidate_budget_bytes is None:
        return {
            "allowed": False, "status": "blocked_kaggle_download_approval_required",
            "reason": (
                "Download Kaggle non autorizzato: specificare esplicitamente "
                "--kaggle-max-download-mib. Non viene selezionata "
                "una quantizzazione inferiore."
            ),
        }
    if len(shard_sizes) != len(shard_names) or any(
        not isinstance(shard_sizes[i], dict)
        or not isinstance(shard_sizes[i].get("bytes"), int)
        or isinstance(shard_sizes[i]["bytes"], bool)
        or shard_sizes[i]["bytes"] <= 0
        for i, _ in missing
    ):
        return {
            "allowed": False, "status": "blocked_kaggle_unknown_transfer_size",
            "reason": (
                "Dimensioni catalogo Kaggle mancanti o non verificabili per "
                "gli shard da scaricare: interrompo prima del download."
            ),
        }
    expected = sum(shard_sizes[i]["bytes"] for i, _ in missing)
    estimate_mib = round(expected / MIB, 2)
    if expected > candidate_budget_bytes:
        return {
            "allowed": False, "status": "blocked_kaggle_download_budget",
            "catalog_transfer_bytes": expected,
            "missing_shards": len(missing),
            "reason": (
                f"Download catalogo ~{estimate_mib} MiB oltre il budget "
                f"residuo {round(candidate_budget_bytes / MIB, 2)} MiB. "
                "La candidata resta non verificata; nessun salto di qualita'."
            ),
        }

    if cache_root is None:
        cache_root = Path(os.environ.get(
            "KAGGLEHUB_CACHE", str(Path.home() / ".cache" / "kagglehub")
        ))
    cache_mount = _existing_ancestor(Path(cache_root))
    target_mount = _existing_ancestor(target_dir)
    reserve = min_free_mib * MIB
    # Allow for rounded display sizes. This is a planning estimate, never
    # a strict network-byte cap because kagglehub controls the transfer.
    rounded = (expected * 105 + 99) // 100
    target_free = shutil.disk_usage(target_mount).free
    cache_free = shutil.disk_usage(cache_mount).free
    same_volume = target_mount.stat().st_dev == cache_mount.stat().st_dev
    required_target = (2 if same_volume else 1) * rounded + reserve
    required_cache = rounded + reserve
    if target_free < required_target or (
        not same_volume and cache_free < required_cache
    ):
        return {
            "allowed": False, "status": "blocked_kaggle_insufficient_disk",
            "catalog_transfer_bytes": expected,
            "missing_shards": len(missing),
            "same_disk_for_cache_and_copy": same_volume,
            "reason": (
                "Spazio disco insufficiente per cache KaggleHub e copia "
                f"locale: download stimato {estimate_mib} MiB, "
                f"riserva richiesta {min_free_mib} MiB. "
                "Nessun download avviato."
            ),
        }
    return {
        "allowed": True,
        "catalog_transfer_bytes": expected,
        "missing_shards": len(missing),
        "catalog_transfer_mib": estimate_mib,
        "candidate_budget_unused_bytes": candidate_budget_bytes - expected,
        "same_disk_for_cache_and_copy": same_volume,
        "free_space_reserve_mib": min_free_mib,
        "note": (
            "Dimensioni dichiarate dal catalogo, possibilmente arrotondate; "
            "approvazione preventiva, non limite rigoroso dei byte trasferiti "
            "ne' misura dei pesi residenti CUDA."
        ),
    }
