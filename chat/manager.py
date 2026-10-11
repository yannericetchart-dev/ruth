class ChatManager:
    """Gestisce una conversazione in RAM, indipendentemente dal motore."""

    def __init__(self, inference):
        self.inference = inference
        self._history: list[dict[str, str]] = []

    def history(self) -> list[dict[str, str]]:
        """Restituisce una copia della cronologia corrente."""
        return [message.copy() for message in self._history]

    def clear(self) -> None:
        """Inizia una nuova conversazione."""
        self._history.clear()

    def send(
        self,
        text: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Invia un messaggio e restituisce la risposta completa."""
        messages = self._messages_with(text)
        answer = self.inference.generate(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._history.extend([
            messages[-1],
            {"role": "assistant", "content": answer},
        ])
        return answer

    def stream(
        self,
        text: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """Invia un messaggio e produce la risposta in streaming."""
        messages = self._messages_with(text)
        parts = []
        for part in self.inference.stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            parts.append(part)
            yield part

        # Salva il turno solo dopo la generazione completa.
        self._history.extend([
            messages[-1],
            {"role": "assistant", "content": "".join(parts)},
        ])

    def _messages_with(self, text: str) -> list[dict[str, str]]:
        content = text.strip()
        if not content:
            raise ValueError("Il messaggio non può essere vuoto.")

        # Gemma 2 IT usa i ruoli user e assistant, senza system separato.
        return self.history() + [{"role": "user", "content": content}]
