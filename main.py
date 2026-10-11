import os

from chat import ChatManager
from hardware import GPUManager
from inference import InferenceManager
from model import ModelManager


def main() -> None:
    gpu = GPUManager.select("nvidia")
    if gpu is None:
        print("Ruth richiede una GPU NVIDIA con driver funzionante e VRAM dedicata.")
        print("Controlla nvidia-smi. Altri produttori saranno supportati in futuro.")
        return

    print(f"GPU: {gpu['name']}")
    print(f"VRAM: {gpu['vram_total_mb']} MiB (liberi: {gpu['vram_free_mb']} MiB)")

    models = ModelManager()
    model_path = models.get()
    if model_path is None:
        print("Modello Gemma 2 9B IT non installato.")
        print('Scaricalo con: .venv/bin/python -c "from model import ModelManager; print(ModelManager().download())"')
        return

    # 6 GB di VRAM non bastano per tutti i layer di Gemma 2 9B più la KV cache.
    # L'offload resta CUDA; i layer rimanenti lavorano in RAM/CPU.
    gpu_layers = int(os.environ.get("RUTH_GPU_LAYERS", "20"))
    inference = InferenceManager(
        engine="llama.cpp",
        backend="auto",
        n_ctx=4096,
        n_gpu_layers=gpu_layers,
    )

    try:
        print(f"Caricamento modello: {model_path.name} (layer GPU richiesti: {gpu_layers})")
        inference.load(model_path)
        print(f"Inferenza: {inference.engine} / {inference.active_backend}")
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
