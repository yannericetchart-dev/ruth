import os
from pathlib import Path
from typing import Iterator

from hardware import GPUManager


class InferenceManager:
    """Inferenza in-process. Al momento: llama.cpp con backend NVIDIA CUDA."""

    ENGINE_BACKENDS = {"llama.cpp": {"nvidia": "cuda"}}

    def __init__(
        self,
        engine: str = "llama.cpp",
        backend: str = "auto",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
    ):
        if engine not in self.ENGINE_BACKENDS:
            raise ValueError(f"Motore di inferenza non supportato: {engine}")
        if backend != "auto" and backend not in self.ENGINE_BACKENDS[engine].values():
            raise ValueError(f"Backend {backend!r} non supportato da {engine}")
        if n_gpu_layers == 0 or n_gpu_layers < -1:
            raise ValueError("Per CUDA, n_gpu_layers deve essere -1 o maggiore di zero.")

        self.engine = engine
        self.backend = backend
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.gpu = None
        self.active_backend = None
        self._llm = None

    def available_backends(self) -> list[str]:
        """Backend utilizzabili sia dall'hardware sia dal binding Python installato."""
        candidates = self.ENGINE_BACKENDS[self.engine]
        if not GPUManager.select("nvidia") or "nvidia" not in candidates:
            return []
        try:
            from llama_cpp.llama_cpp import llama_supports_gpu_offload
            if llama_supports_gpu_offload():
                return [candidates["nvidia"]]
        except (ImportError, OSError, AttributeError):
            pass
        return []

    def load(self, model_path: str | Path) -> None:
        """Carica il modello sulla GPU NVIDIA selezionata."""
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Modello non trovato: {path}")

        gpu = GPUManager.select("nvidia")
        if gpu is None:
            detected = [item["vendor"] for item in GPUManager.detect()]
            raise RuntimeError(
                f"GPU NVIDIA con VRAM accessibile non trovata (rilevate: {detected}). "
                "Verifica nvidia-smi e il driver NVIDIA."
            )

        selected = self.ENGINE_BACKENDS[self.engine]["nvidia"]
        if self.backend not in ("auto", selected):
            raise RuntimeError(f"Backend {self.backend} non disponibile per {self.engine}.")

        # La variabile deve essere impostata prima del caricamento del runtime CUDA.
        # Una configurazione esplicita dell'utente ha priorità.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu["uuid"])

        if selected not in self.available_backends():
            raise RuntimeError(
                "Il binding llama-cpp-python nel venv non supporta l'offload GPU. "
                "Esegui ./setup_env.sh per compilare/abilitare CUDA."
            )

        from llama_cpp import Llama

        self.close()
        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        self.gpu = gpu
        self.active_backend = selected

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        result = self._model().create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return result["choices"][0]["message"].get("content") or ""

    def stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        chunks = self._model().create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in chunks:
            content = chunk["choices"][0]["delta"].get("content")
            if content:
                yield content

    def close(self) -> None:
        if self._llm is not None:
            llm = self._llm
            self._llm = None
            llm.close()
        self.gpu = None
        self.active_backend = None

    def _model(self):
        if self._llm is None:
            raise RuntimeError("Nessun modello caricato: esegui load() prima.")
        return self._llm
