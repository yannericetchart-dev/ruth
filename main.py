from chat import ChatManager
from inference import InferenceManager
from model import ModelManager


def main() -> None:
    models = ModelManager()
    model_path = models.get()

    if model_path is None:
        print("Modello Gemma 2 9B IT non installato.")
        print('Scaricalo con: .venv/bin/python -c "from model import ModelManager; print(ModelManager().download())"')
        return

    inference = InferenceManager(n_ctx=4096, n_gpu_layers=0)

    try:
        print(f"Caricamento modello: {model_path.name}")
        inference.load(model_path)
        chat = ChatManager(inference)

        print("Ruth avviata. Comandi: /clear, /exit")

        while True:
            try:
                message = input("\nTu: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nArrivederci!")
                break

            if not message:
                continue
            if message.lower() == "/exit":
                print("Arrivederci!")
                break
            if message.lower() == "/clear":
                chat.clear()
                print("Conversazione azzerata.")
                continue

            print("Ruth: ", end="", flush=True)
            try:
                for chunk in chat.stream(message):
                    print(chunk, end="", flush=True)
                print()
            except KeyboardInterrupt:
                print("\nGenerazione interrotta. Chiusura di Ruth.")
                break
            except Exception as exc:
                print(f"\nErrore di generazione: {exc}")

    finally:
        inference.close()


if __name__ == "__main__":
    main()
