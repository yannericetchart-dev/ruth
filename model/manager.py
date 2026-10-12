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
        }
    }

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

    def selected_baseline(self, model_id: str = "gemma-2-9b-it") -> dict | None:
        """Read the full-GPU calibration selection, without changing get().

        Only the explicit follow-on profiling stage should promote this
        experimental baseline into Ruth's operational model configuration.
        """
        import json

        base_dir = self.root / model_id
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
        return {**selection, "local_path": str(base_dir / name)}

    def download(self, model_id: str = "gemma-2-9b-it") -> Path:
        """Scarica il GGUF selezionato, soltanto se manca."""
        spec = self._spec(model_id)
        installed = self.get(model_id)
        if installed is not None:
            return installed

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
