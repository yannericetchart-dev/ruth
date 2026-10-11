from pathlib import Path
from typing import Iterator


class InferenceManager:
    """Inferenza GGUF in-process tramite llama-cpp-python."""

    def __init__(self, n_ctx: int = 8192, n_gpu_layers: int = 0):
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None

    def load(self, model_path: str | Path) -> None:
        """Carica un GGUF locale in memoria."""
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Modello non trovato: {path}")

        from llama_cpp import Llama

        self.close()
        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Restituisce la risposta completa come stringa."""
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
        """Produce frammenti di testo man mano che vengono generati."""
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
        """Libera le risorse native del modello caricato."""
        if self._llm is not None:
            llm = self._llm
            self._llm = None
            llm.close()

    def _model(self):
        if self._llm is None:
            raise RuntimeError("Nessun modello caricato: esegui load() prima.")
        return self._llm
