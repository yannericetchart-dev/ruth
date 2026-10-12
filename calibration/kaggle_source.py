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
    if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
        raise ValueError(
            "Kaggle richiede owner/model/framework/variation[/version]."
        )
    if "/".join(handle.split("/")[:2]) != repository:
        raise ValueError("La variante Kaggle non appartiene al modello selezionato.")
    return handle


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
        # Do not echo CLI stderr: it can include credentials/auth fragments.
        raise RuntimeError(
            "Interrogazione Kaggle fallita (codice di uscita "
            f"{run.returncode}). Verifica il modello, l'accesso e "
            "l'handle della variante con 'kaggle models variations files' "
            "oppure 'kaggle models variations versions files'."
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


def inventory_kaggle(
    repository: str, gguf_prefix: str,
    selected_handle: str | None = None,
) -> list[dict]:
    """List GGUF files actually returned by Kaggle, never invented variants."""
    candidates = []
    handles = variation_handles(repository, selected_handle)
    for handle in handles:
        # A pinned version uses a different official Kaggle CLI subcommand.
        if len(handle.split("/")) == 5:
            rows = _csv_call("variations", "versions", "files", handle,
                             "--page-size", "100")
        else:
            rows = _csv_call("variations", "files", handle,
                             "--page-size", "100")
        for row in rows:
            filename = _filename(row)
            if not filename.casefold().endswith(".gguf"):
                continue
            base = Path(filename).name
            if not base.casefold().startswith(gguf_prefix.casefold()):
                # Do not confuse another derivative with the chosen model.
                continue
            quant = base[len(gguf_prefix):-5]
            ranked = quantization_priority(quant)
            candidates.append({
                "repo_id": repository, "filename": base,
                "source_file_path": filename,
                "quantization": quant, **ranked,
                "kaggle_handle": handle,
                "file_bytes": None,  # Kaggle does not expose GGUF range header.
                "cuda_baseline_eligible": True,
                "weight_size_source": "unknown_until_local_gguf_download",
                "ranking_basis": "same_quantization_family_prior_as_huggingface",
                "source": "kaggle",
            })
    if not candidates:
        raise RuntimeError(
            "Nessun file .gguf di Mistral 7B Instruct v0.3 presente nelle "
            "varianti Kaggle interrogate. Nessun fallback a Hugging Face."
        )
    # Do not silently choose between same-named files on different variants.
    duplicates = {}
    for c in candidates:
        duplicates.setdefault(c["filename"].casefold(), []).append(c)
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
