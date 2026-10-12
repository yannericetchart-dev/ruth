from pathlib import Path


class ModelManager:
    """Catalogo e download locale dei modelli di Ruth."""

    CATALOG = {
        "gemma-2-9b-it": {
            "source": "bartowski",
            "repository": "bartowski/gemma-2-9b-it-GGUF",
            "filename": "gemma-2-9b-it-Q4_K_M.gguf",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "gguf_prefix": "gemma-2-9b-it-",
            "architecture": "gemma2",
        },
        "qwen-2.5-7b-instruct": {
            "source": "bartowski",
            "repository": "bartowski/Qwen2.5-7B-Instruct-GGUF",
            "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "gguf_prefix": "Qwen2.5-7B-Instruct-",
            "architecture": "qwen2",
        },
        "mistral-7b-instruct-v0.3": {
            "source": "bartowski",
            "repository": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "filename": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "gguf_prefix": "Mistral-7B-Instruct-v0.3-",
            "architecture": "llama",
            "chat_template": "mistral_inst",
        },
        "llama-3.1-8b-instruct": {
            "source": "bartowski",
            "repository": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "format": "gguf",
            "quantization": "Q4_K_M",
            "gguf_prefix": "Meta-Llama-3.1-8B-Instruct-",
            "architecture": "llama",
        }
    }

    # Optional alternative publishers; Hugging Face remains the default.
    SOURCE_OVERRIDES = {
        "gemma-2-9b-it": {
            "kaggle": {
                "repository": "google/gemma-2",
                "kaggle_handle": "google/gemma-2/gguf/2.0-9b-it",
                "access_requires_consent": True,
                "access_page": "https://www.kaggle.com/models/google/gemma-2",
            },
        },
        "qwen-2.5-7b-instruct": {
            "kaggle": {
                "repository": "qwen-lm/qwen2.5",
                "kaggle_handle": "qwen-lm/qwen2.5/gguf/7b-instruct",
            },
        },
    }

    @classmethod
    def available_sources(cls, model_id: str) -> tuple[str, ...]:
        if model_id not in cls.CATALOG:
            raise ValueError(f"Modello sconosciuto: {model_id}")
        return (cls.CATALOG[model_id]["source"],
                *cls.SOURCE_OVERRIDES.get(model_id, {}))

    @classmethod
    def source_spec(cls, model_id: str, source: str | None = None) -> dict:
        if model_id not in cls.CATALOG:
            raise ValueError(f"Modello sconosciuto: {model_id}")
        default = cls.CATALOG[model_id]
        source = source or default["source"]
        if source == default["source"]:
            return dict(default)
        override = cls.SOURCE_OVERRIDES.get(model_id, {}).get(source)
        if override is None:
            raise ValueError(
                f"Fonte '{source}' non disponibile per {model_id}; "
                f"disponibili: {', '.join(cls.available_sources(model_id))}."
            )
        return {**default, **override, "source": source}

    def model_directory(self, model_id: str, source: str | None = None,
                        repository: str | None = None) -> Path:
        spec = self.source_spec(model_id, source)
        repo = repository or spec["repository"]
        if spec["source"] == self.CATALOG[model_id]["source"]:
            if repo != spec["repository"]:
                raise ValueError("Repository differente dalla fonte predefinita.")
            return self.root / model_id
        from calibration.kaggle_source import validate_handle
        if "/" not in repo or len(repo.split("/")) != 2:
            raise ValueError("Repository Kaggle non valido.")
        validate_handle(repo + "/gguf/placeholder", repo)
        return self.root / model_id / "sources" / spec["source"] / repo

    def __init__(self, root: str | Path | None = None):
        self.root = (
            Path(root).expanduser().resolve()
            if root is not None
            else Path(__file__).resolve().parents[1] / "models"
        )

    def list(self) -> list[dict]:
        """Modelli disponibili nel catalogo, con stato locale."""
        return [
            {"id": model_id, **spec, "installed": self.get(model_id) is not None}
            for model_id, spec in self.CATALOG.items()
        ]

    def get(self, model_id: str = "gemma-2-9b-it") -> Path | None:
        """Percorso del GGUF, oppure None se non è installato."""
        spec = self._spec(model_id)
        path = self.root / model_id / spec["filename"]
        return path if path.is_file() else None

    def selected_baseline(self, model_id: str = "gemma-2-9b-it", *,
                          source: str | None = None,
                          repository: str | None = None) -> dict | None:
        """Read the full-GPU calibration selection, without changing get().

        Only the explicit follow-on profiling stage should promote this
        experimental baseline into Ruth's operational model configuration.
        """
        import json

        base_dir = self.model_directory(model_id, source, repository)
        manifest = base_dir / "baseline_selection.json"
        if not manifest.is_file():
            return None
        selection = json.loads(manifest.read_text())
        name = selection.get("filename")
        if (not selection.get("native_verified_all_gpu")
            or not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".gguf")
            or not (base_dir / name).is_file()):
            raise ValueError("Baseline persistita incompleta o non verificabile.")
        expected_files = selection.get("shard_filenames") or [name]
        if (not isinstance(expected_files, list)
            or not expected_files or expected_files[0] != name
            or any(not isinstance(filename, str)
                   or Path(filename).name != filename
                   or not (base_dir / filename).is_file()
                   for filename in expected_files)):
            raise ValueError("Baseline multi-shard incompleta o non verificabile.")
        expected = self.source_spec(model_id, source)
        if (selection.get("source", expected["source"]) != expected["source"]
            or selection.get("repository") !=
                (repository or expected["repository"])
            or selection.get("model_id") != model_id):
            raise ValueError("Baseline di una fonte/repository diverso.")
        return {**selection, "local_path": str(base_dir / name)}

    def download(self, model_id: str = "gemma-2-9b-it") -> Path:
        """Scarica il GGUF selezionato, soltanto se manca."""
        spec = self._spec(model_id)
        installed = self.get(model_id)
        if installed is not None:
            return installed

        if spec["source"] == "kaggle":
            raise RuntimeError(
                "Per un modello Kaggle, utilizzare "
                "calibration.profile_pipeline: seleziona prima "
                "il GGUF disponibile nella variante Kaggle."
            )
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=spec["repository"],
                filename=spec["filename"],
                local_dir=self.root / model_id,
            )
        )

    def inspect(self, model_id: str = "gemma-2-9b-it", n_ctx: int = 4096) -> dict:
        """Struttura GGUF in sola lettura, senza eseguire inferenza."""
        from .inspect import inspect_gguf

        path = self.get(model_id)
        if path is None:
            raise FileNotFoundError(f"Modello non installato: {model_id}")
        return inspect_gguf(path, n_ctx=n_ctx)

    def _spec(self, model_id: str) -> dict:
        try:
            return self.CATALOG[model_id]
        except KeyError:
            raise ValueError(f"Modello sconosciuto: {model_id}") from None
