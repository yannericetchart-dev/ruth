"""Kaggle GGUF source adapter: native Kaggle catalog and downloads, NO HF fallback.

Kaggle paths are model/variation/version, unlike Hugging Face repositories.
Use the official Kaggle CLI for discovery and kagglehub for single-file
downloads. A model page alone does not guarantee that a GGUF variation exists.
"""

import csv
import io
from pathlib import Path
import re
import shutil
import subprocess
import sys

from .gguf_catalog import quantization_priority


_HANDLE = re.compile(
    r"^[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
    r"(?:/[1-9]\d*)?$"
)


def validate_handle(handle: str, repository: str) -> str:
    """Accept a real variation handle, never an unchecked model-only path."""
    if (not isinstance(handle, str) or not _HANDLE.fullmatch(handle)
        or any(part in (".", "..") for part in handle.split("/"))):
        raise ValueError(
            "Kaggle richiede owner/model/framework/variation[/version] "
            "senza segmenti di percorso relativi."
        )
    if "/".join(handle.split("/")[:2]) != repository:
        raise ValueError("La variante Kaggle non appartiene al modello selezionato.")
    return handle


def classify_kaggle_cli_failure(stderr: str) -> tuple[str, str]:
    """Return a safe, stable error category; never include upstream stderr.

    Kaggle CLI error text can contain tokens, request URLs or auth headers.
    Only recognize fixed phrases and HTTP codes; keep everything else private.
    """
    details = stderr.casefold()
    # Distinguish absent credentials from an absent MODEL. The old broad
    # "not found" check incorrectly reported both as missing variations.
    if re.search(
        r"(?:credentials?|api[_ ]?tokens?|api[_ ]?keys?)"
        r".{0,70}(?:not found|missing|not configured|unavailable)"
        r"|(?:not found|missing).{0,70}"
        r"(?:credentials?|api[_ ]?tokens?|api[_ ]?keys?)"
        r"|authentication required|please authenticate", details,
    ):
        return ("credentials_missing",
                "Credenziali Kaggle non disponibili per la CLI. "
                "Verifica l'accesso con 'kaggle quota', senza stampare token.")
    if re.search(r"\b401\b|unauthorized|unauthenticated", details):
        return ("unauthorized",
                "Autenticazione rifiutata; controlla il token Kaggle attivo.")
    if re.search(r"\b403\b|forbidden|permission denied|access denied", details):
        return ("forbidden",
                "Accesso vietato: controlla consenso/licenza e permessi "
                "sulla variante Kaggle.")
    if re.search(
        r"\b404\b"
        r"|(?:model|variation|version|resource).{0,60}not found"
        r"|not found.{0,60}(?:model|variation|version|resource)",
        details,
    ):
        return ("not_found",
                "Risorsa modello/variante non trovata oppure non accessibile "
                "con questo account. Verifica l'handle e l'accesso.")
    if re.search(r"\b429\b|rate limit|too many requests", details):
        return ("rate_limited",
                "Limite richieste Kaggle; riprova più tardi.")
    if re.search(r"\b5(?:00|02|03|04)\b|internal server error", details):
        return ("server_error",
                "Servizio Kaggle temporaneamente non disponibile.")
    if re.search(r"invalid choice|unrecognized arguments|usage:|no such option",
                 details):
        return ("cli_syntax",
                "Versione CLI Kaggle incompatibile con il comando richiesto.")
    if re.search(r"sslerror|connectionerror|name resolution|timed out", details):
        return ("network",
                "Problema di rete/TLS nel collegamento con Kaggle.")
    return ("unknown",
            "Causa non identificabile dal messaggio filtrato. "
            "Verifica la CLI e la risposta del servizio in locale.")


def _csv_call(*args: str) -> list[dict]:
    try:
        # Prefer the CLI installed in Ruth's venv even when not activated.
        venv_cli = Path(sys.executable).with_name("kaggle")
        cli = str(venv_cli) if venv_cli.is_file() else "kaggle"
        run = subprocess.run(
            [cli, "models", *args, "--csv"],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "CLI Kaggle mancante: installare i pacchetti kaggle e kagglehub."
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout interrogazione catalogo Kaggle.") from None
    if run.returncode != 0:
        # Only expose recognized error categories, never raw stderr/stdout.
        cause, hint = classify_kaggle_cli_failure(run.stderr or "")
        raise RuntimeError(
            f"Interrogazione Kaggle fallita (exit={run.returncode}, "
            f"categoria={cause}). {hint} "
            "Comando di verifica: kaggle models variations files "
            "<owner/model/gguf/variation> --csv."
        )
    content = run.stdout.strip()
    if not content:
        return []
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        raise RuntimeError("Kaggle ha restituito dati CSV senza intestazioni.")
    return list(reader)


def _get(record: dict, *keys: str) -> str:
    mapping = {
        re.sub(r"[^a-z0-9]", "", str(k).casefold()): str(v).strip()
        for k, v in record.items() if k is not None and v is not None
    }
    for key in keys:
        value = mapping.get(re.sub(r"[^a-z0-9]", "", key.casefold()))
        if value:
            return value
    return ""


def variation_handles(repository: str, selected_handle: str | None = None) -> list[str]:
    if selected_handle:
        return [validate_handle(selected_handle, repository)]
    # Kaggle CLI 2.2.4 has no "models variations list" command.
    # Never invent a variation or interpret an unsupported command as auth denial.
    raise RuntimeError(
        "La CLI Kaggle non supporta la ricerca automatica delle varianti "
        "(comando 'models variations list' inesistente). Apri la pagina "
        f"https://www.kaggle.com/models/{repository}, verifica una variante "
        "che contenga file GGUF e specifica il suo handle reale con "
        "--kaggle-handle owner/model/framework/variation[/version]. "
        "Nessun fallback implicito a Hugging Face."
    )


def _filename(record: dict) -> str:
    value = _get(record, "name", "fileName", "filename", "filePath", "path")
    return value


def variation_files(handle: str) -> tuple[list[dict], str]:
    """List files without downloading bytes; pin any discovered version.

    The latest-version endpoint may return 404 while the versioned endpoint
    works. Only retry a version explicitly returned by Kaggle's catalog.
    If version discovery is also inaccessible, retain the original failure.
    """
    if len(handle.split("/")) == 5:
        return (_csv_call("variations", "versions", "files", handle,
                          "--page-size", "100"), handle)
    try:
        return (_csv_call("variations", "files", handle,
                          "--page-size", "100"), handle)
    except RuntimeError as original:
        if "categoria=not_found" not in str(original):
            raise
        try:
            versions = _csv_call("variations", "versions", "list", handle,
                                 "--page-size", "100")
        except RuntimeError as secondary:
            # Both errors are already sanitized by _csv_call.
            raise RuntimeError(
                f"{original} Ricerca delle versioni: {secondary}"
            ) from None
        numeric_versions = set()
        for version in versions:
            value = _get(version, "versionNumber", "version", "number")
            if re.fullmatch(r"[1-9]\d*", value):
                numeric_versions.add(int(value))
        if not numeric_versions:
            raise RuntimeError(
                f"{original} La risposta Kaggle non contiene numeri "
                "di versione verificabili."
            ) from None
        pinned = f"{handle}/{max(numeric_versions)}"
        try:
            rows = _csv_call("variations", "versions", "files", pinned,
                             "--page-size", "100")
        except RuntimeError:
            raise original from None
        return rows, pinned


_SPLIT_GGUF = re.compile(
    r"^(?P<name>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$",
    re.IGNORECASE,
)


def _gguf_quantization(logical_name: str, gguf_prefix: str) -> tuple[str, bool]:
    """Read the precision from a whole-model name, never from a split suffix."""
    matched = logical_name.casefold().startswith(gguf_prefix.casefold())
    quant = (logical_name[len(gguf_prefix):-5] if matched
             else logical_name[:-5])
    return quant, matched


def inventory_kaggle(
    repository: str, gguf_prefix: str,
    selected_handle: str | None = None,
) -> list[dict]:
    """Produce one logical candidate per COMPLETE sharded GGUF group.

    Retain incomplete groups as unranked blockers: it is unsafe to treat
    a single shard as an independently loadable model. No weight downloads.
    """
    candidates = []
    for handle in variation_handles(repository, selected_handle):
        rows, resolved_handle = variation_files(handle)
        splits = {}
        singles = []
        for row in rows:
            source_path = _filename(row)
            if not source_path.casefold().endswith(".gguf"):
                continue
            filename = Path(source_path).name
            found = _SPLIT_GGUF.fullmatch(filename)
            if found:
                index = int(found["part"])
                total = int(found["total"])
                # Split numbering is 1-based and must never be trusted blindly.
                if not (1 <= index <= total <= 99999):
                    raise RuntimeError("Inventario Kaggle: indice GGUF split non valido.")
                logical_name = found["name"] + ".gguf"
                group = splits.setdefault(logical_name.casefold(), {
                    "logical_name": logical_name, "parts": {}, "total": total,
                })
                if group["total"] != total or index in group["parts"]:
                    raise RuntimeError(
                        "Inventario Kaggle: shard duplicati o totale incoerente."
                    )
                group["parts"][index] = (filename, source_path)
            else:
                singles.append((filename, source_path))

        def new_candidate(
            filename: str, source_path: str, quant: str, matched: bool,
            *, shard_filenames: list[str], shard_paths: list[str],
            split_total: int = 1, split_complete: bool = True,
            missing: list[int] | None = None,
        ) -> dict:
            ranked = (quantization_priority(quant) if matched and split_complete
                      else quantization_priority(""))
            return {
                "repo_id": repository,
                "filename": filename,
                "source_file_path": source_path,
                "filename_matches_model_prefix": matched,
                "quantization": quant, **ranked,
                "kaggle_handle": resolved_handle,
                "shard_filenames": shard_filenames,
                "shard_source_file_paths": shard_paths,
                "shard_count": split_total,
                "split_complete": split_complete,
                "missing_shard_indices": missing or [],
                "file_bytes": None,
                "cuda_baseline_eligible": split_complete,
                "weight_size_source": "unknown_until_local_gguf_download",
                "ranking_basis": "coarse_quantization_family_prior_not_measured",
                "source": "kaggle",
            }

        for filename, source_path in singles:
            quant, matched = _gguf_quantization(filename, gguf_prefix)
            candidates.append(new_candidate(
                filename, source_path, quant, matched,
                shard_filenames=[filename], shard_paths=[source_path],
            ))
        for group in splits.values():
            total = group["total"]
            parts = group["parts"]
            missing = [i for i in range(1, total + 1) if i not in parts]
            present = [parts[i] for i in sorted(parts)]
            logical = group["logical_name"]
            quant, matched = _gguf_quantization(logical, gguf_prefix)
            # A valid split set must begin with shard 1; incomplete sets
            # never proceed to downloads, native parsing, or baseline.
            first_file, first_path = present[0]
            candidates.append(new_candidate(
                first_file, first_path, quant, matched,
                shard_filenames=[pair[0] for pair in present],
                shard_paths=[pair[1] for pair in present],
                split_total=total, split_complete=not missing,
                missing=missing,
            ))

    if not candidates:
        raise RuntimeError(
            "Nessun file .gguf presente nelle "
            "varianti Kaggle interrogate. Nessun fallback a Hugging Face."
        )
    duplicates = {}
    for candidate in candidates:
        duplicates.setdefault(candidate["filename"].casefold(), []).append(candidate)
    ambiguous = [name for name, group in duplicates.items() if len(group) > 1]
    if ambiguous:
        raise RuntimeError(
            f"GGUF duplicati in varianti Kaggle differenti: {ambiguous}. "
            "Specifica un --kaggle-handle preciso."
        )
    return sorted(
        candidates, key=lambda x: (
            x["quality_priority"] is None,
            x["quality_priority"] if x["quality_priority"] is not None else 0,
            x["filename"].casefold(),
        ),
    )


def download_kaggle(candidate: dict, destination: Path) -> Path:
    """Download exactly one Kaggle GGUF and preserve a local source copy."""
    import kagglehub

    filename = candidate["filename"]
    source_file_path = candidate["source_file_path"]
    if (not filename.casefold().endswith(".gguf")
        or Path(source_file_path).name != filename):
        raise ValueError("File GGUF Kaggle non sicuro o inatteso.")
    handle = validate_handle(candidate["kaggle_handle"], candidate["repo_id"])
    downloaded = Path(kagglehub.model_download(handle, path=source_file_path))
    if not downloaded.is_file() or downloaded.name != filename:
        raise RuntimeError("Kaggle non ha restituito il GGUF richiesto.")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / filename
    if not target.is_file():
        partial = target.with_suffix(target.suffix + ".part")
        try:
            shutil.copyfile(downloaded, partial)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    return target
