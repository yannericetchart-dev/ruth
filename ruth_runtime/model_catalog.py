"""Ruth hierarchical model catalog and explicitly authorized HF metadata resolver.

No dependencies outside Python stdlib, model downloads, license acceptance,
inference, pip installs or changes to the CUDA runtime. All model artifacts
require pinned revision and SHA256 before a separate download workflow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from .dependencies import PROJECT_ROOT

CATALOG_FILE = Path("catalog/model_catalog.json")
SHA256 = re.compile(r"^[a-f0-9]{64}$")
GIT_SHA = re.compile(r"^[a-f0-9]{40}$")
SEGMENT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,99}$")
REPO_ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$")
VALID_PURPOSES = frozenset({"production", "test"})


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class ModelChoice:
    purpose: str
    family: str
    model: str
    version: str
    format: str
    quantization: str
    path: str
    source: str
    repo_id: str
    filename: str
    upstream_repo: str
    license: str
    license_acknowledgement_required: bool
    expected_sha256: str
    approx_size_mb: int
    expected_size_bytes: int | None
    revision: str | None
    assessment: str
    local_model_ready: bool
    downloadable: bool
    model_loaded: bool
    gpu_offload_executed: bool
    inference_executed: bool

    def to_dict(self) -> dict:
        return asdict(self)


def load_catalog(project_root: Path = PROJECT_ROOT) -> dict:
    path = Path(project_root) / CATALOG_FILE
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise CatalogError("Cannot read versioned Ruth model catalog") from exc
    if not isinstance(catalog, dict) or catalog.get("schema") != "ruth-model-catalog-v1":
        raise CatalogError("Unsupported catalog schema")
    if catalog.get("default_source") != "huggingface":
        raise CatalogError("Default source must be Hugging Face")
    sources = catalog.get("sources", {})
    if sources.get("huggingface", {}).get("status") != "implemented" or \
            sources.get("kaggle", {}).get("status") != "TODO":
        raise CatalogError("Hugging Face/Kaggle provider policy changed")
    if catalog.get("families") != ["gemma", "llama", "qwen", "mistral"]:
        raise CatalogError("Unexpected initial model families")
    entries = catalog.get("models")
    if not isinstance(entries, list) or not entries:
        raise CatalogError("Catalog contains no usable models")
    seen: set[tuple[str, ...]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise CatalogError("Invalid model record")
        for key in ("family", "model", "version", "format"):
            if not isinstance(entry.get(key), str) or not SEGMENT.fullmatch(entry[key]):
                raise CatalogError("Unsafe or invalid model identifier")
        if entry["family"] not in catalog["families"]:
            raise CatalogError("Unknown family")
        if entry["format"] != "gguf":
            raise CatalogError("Only GGUF candidates are supported")
        if not isinstance(entry.get("purpose"), list) or \
                not set(entry["purpose"]).issubset(VALID_PURPOSES):
            raise CatalogError("Invalid purpose mapping")
        if not entry.get("license_acknowledgement_required") \
                or not isinstance(entry.get("license"), str):
            raise CatalogError("Explicit model license review is mandatory")
        if not REPO_ID.fullmatch(entry.get("upstream_repo", "")):
            raise CatalogError("Invalid upstream model repository")
        variants = entry.get("variants")
        if not isinstance(variants, list) or not variants:
            raise CatalogError("No pinned GGUF variants")
        for item in variants:
            if not isinstance(item, dict) or item.get("source") != "huggingface" \
                    or not REPO_ID.fullmatch(item.get("repo_id", "")):
                raise CatalogError("Only explicit Hugging Face source is supported")
            name = item.get("filename", "")
            if not isinstance(name, str) or Path(name).name != name \
                    or not name.endswith(".gguf"):
                raise CatalogError("Invalid GGUF filename")
            sha = item.get("expected_sha256")
            if not isinstance(sha, str) or not SHA256.fullmatch(sha):
                raise CatalogError("Missing pinned GGUF SHA256")
            quant = item.get("quantization")
            if not isinstance(quant, str) or not SEGMENT.fullmatch(quant):
                raise CatalogError("Invalid quantization identifier")
            exact_size = item.get("expected_size_bytes")
            if exact_size is not None and (type(exact_size) is not int or
                                           exact_size < 1024 * 1024):
                raise CatalogError("Invalid exact GGUF size pin")
            revision = item.get("revision")
            if revision is not None and (not isinstance(revision, str) or
                                         not GIT_SHA.fullmatch(revision)):
                raise CatalogError("Only immutable Git SHA revisions supported")
            if type(item.get("approx_size_mb")) is not int \
                    or item["approx_size_mb"] <= 0:
                raise CatalogError("Invalid size estimate")
            ident = (entry["family"], entry["model"], entry["version"],
                     entry["format"], quant)
            if ident in seen:
                raise CatalogError("Duplicate catalog model variant")
            seen.add(ident)
    for purpose, ident in catalog.get("defaults", {}).items():
        if purpose not in VALID_PURPOSES or not isinstance(ident, dict):
            raise CatalogError("Unexpected default purpose")
        if tuple(ident.get(k) for k in
                 ("family", "model", "version", "format", "quantization")) not in seen:
            raise CatalogError("Default points to a missing catalog entry")
    if set(catalog["defaults"]) != VALID_PURPOSES:
        raise CatalogError("Production and test defaults required")
    return catalog


def list_models(project_root: Path = PROJECT_ROOT) -> dict:
    catalog = load_catalog(project_root)
    return {
        "status": "catalog_ready",
        "schema": catalog["schema"],
        "default_source": catalog["default_source"],
        "sources": catalog["sources"],
        "families": [
            {
                "family": family,
                "models": [
                    {
                        "model": entry["model"], "version": entry["version"],
                        "format": entry["format"],
                        "quantizations": [v["quantization"] for v in entry["variants"]],
                        "source": "huggingface",
                        "license": entry["license"],
                    }
                    for entry in catalog["models"] if entry["family"] == family
                ],
                "status": "candidate_available"
                if any(e["family"] == family for e in catalog["models"])
                else "registered_no_verified_candidate",
            }
            for family in catalog["families"]
        ],
        "defaults": catalog["defaults"],
        "download_started": False,
        "gpu_inference_verified": False,
    }


def select_model(
    purpose: str = "test", *,
    family: str | None = None,
    model: str | None = None,
    version: str | None = None,
    format: str | None = None,
    quantization: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> ModelChoice:
    if purpose not in VALID_PURPOSES:
        raise CatalogError("Purpose must be 'test' or 'production'")
    catalog = load_catalog(project_root)
    defaults = catalog["defaults"][purpose]
    selected = {
        key: (value if value is not None else defaults[key])
        for key, value in {
            "family": family, "model": model, "version": version,
            "format": format, "quantization": quantization
        }.items()
    }
    # Explicit overrides are exact, never silently substitute another family,
    # size, quantization, or a CPU-capable wheel.
    for entry in catalog["models"]:
        if all(entry[k] == selected[k] for k in
               ("family", "model", "version", "format")):
            for item in entry["variants"]:
                if item["quantization"] == selected["quantization"]:
                    return ModelChoice(
                        purpose=purpose,
                        **{k: selected[k] for k in
                           ("family", "model", "version", "format",
                            "quantization")},
                        path="/".join(selected[k] for k in
                                      ("family", "model", "version", "format",
                                       "quantization")),
                        source=item["source"], repo_id=item["repo_id"],
                        filename=item["filename"],
                        upstream_repo=entry["upstream_repo"],
                        license=entry["license"],
                        license_acknowledgement_required=True,
                        expected_sha256=item["expected_sha256"],
                        approx_size_mb=item["approx_size_mb"],
                        expected_size_bytes=item.get("expected_size_bytes"),
                        revision=item["revision"],
                        assessment=item.get("assessment", "not_gpu_qualified"),
                        local_model_ready=False, downloadable=False,
                        model_loaded=False, gpu_offload_executed=False,
                        inference_executed=False,
                    )
    raise CatalogError("Model path/quantization not registered; no automatic fallback")


def fetch_huggingface_metadata(repo_id: str, *, revision: str = "main") -> dict:
    """Read only public HF metadata. Never follow a redirect off Hugging Face."""
    if not REPO_ID.fullmatch(repo_id):
        raise CatalogError("Invalid Hugging Face repository ID")
    if revision != "main" and not GIT_SHA.fullmatch(revision):
        raise CatalogError("Hugging Face metadata revision must be immutable SHA")
    url = ("https://huggingface.co/api/models/" +
           "/".join(quote(x, safe="") for x in repo_id.split("/")) +
           "/revision/" + quote(revision, safe="") + "?blobs=true")
    req = Request(url, headers={"User-Agent": "Ruth-model-catalog/1"})
    with urlopen(req, timeout=20) as response:
        if urlsplit(response.url).hostname != "huggingface.co" \
                or urlsplit(response.url).scheme != "https":
            raise CatalogError("Metadata redirect outside trusted HF API")
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise CatalogError("HF metadata response too large")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise CatalogError("Malformed HF metadata response")
    return data


def resolve_model_metadata(
    choice: ModelChoice,
    *, authorize_metadata_network: bool = False,
    fetch: Callable[[str], dict] | None = None,
) -> dict:
    """Pin immutable HF commit and GGUF LFS digest; NEVER downloads weights."""
    base = {
        "status": "metadata_network_not_authorized",
        "selection": choice.to_dict(),
        "download_started": False, "license_accepted": False,
        "model_loaded": False, "gpu_offload_executed": False,
        "inference_executed": False,
    }
    if not authorize_metadata_network:
        return base
    try:
        # Immutable pin: the author's default branch may have moved since
        # this GGUF's checksum and revision were approved.
        info = (fetch(choice.repo_id) if fetch is not None else
                fetch_huggingface_metadata(
                    choice.repo_id, revision=choice.revision or "main"))
        if not isinstance(info, dict):
            raise CatalogError("Invalid metadata type")
        revision = info.get("sha")
        if not isinstance(revision, str) or not GIT_SHA.fullmatch(revision):
            raise CatalogError("No immutable 40-char repository revision")
        if choice.revision is not None and revision != choice.revision:
            raise CatalogError("Upstream revision changed from catalog pin")
        siblings = info.get("siblings")
        if not isinstance(siblings, list):
            raise CatalogError("HF response missing siblings")
        matches = [s for s in siblings if
                   isinstance(s, dict) and s.get("rfilename") == choice.filename]
        if len(matches) != 1:
            raise CatalogError("Exact GGUF filename absent or ambiguous")
        entry = matches[0]
        lfs = entry.get("lfs")
        if not isinstance(lfs, dict):
            raise CatalogError("GGUF LFS SHA256 metadata not available")
        observed_hash = lfs.get("sha256")
        if observed_hash != choice.expected_sha256:
            raise CatalogError("Remote GGUF SHA256 differs from catalog pin")
        size = entry.get("size", lfs.get("size"))
        if type(size) is not int or size < 1024 * 1024:
            raise CatalogError("GGUF size missing or invalid")
        if choice.expected_size_bytes is not None and size != choice.expected_size_bytes:
            raise CatalogError("Remote GGUF size differs from catalog pin")
        if info.get("private") is True:
            raise CatalogError("Private repository requires separate authentication")
        license_reported = None
        if isinstance(info.get("cardData"), dict):
            license_reported = info["cardData"].get("license")
        if license_reported is not None and license_reported != choice.license:
            raise CatalogError("Model license changed")
        return {
            **base, "status": "source_metadata_verified_license_pending",
            "repo_id": choice.repo_id, "revision": revision,
            "filename": choice.filename,
            "sha256": observed_hash, "size_bytes": size,
            "gated": info.get("gated", "unknown"),
            "license_expected": choice.license,
            "license_reported": license_reported,
            "model_url": ("https://huggingface.co/" + choice.repo_id +
                          "/blob/" + revision + "/" + quote(choice.filename)),
            "download_started": False,
        }
    except (OSError, ValueError, TypeError, KeyError,
            json.JSONDecodeError) as exc:
        return {**base, "status": "source_metadata_unverified",
                "error": type(exc).__name__ + ": " + str(exc)[:260]}
