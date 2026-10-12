# Ruth

Assistente sperimentale Python/Linux con backend `llama-cpp-python` e CUDA NVIDIA.

## Installazione

```bash
git clone -b feature/minimal-chat https://github.com/yannericetchart-dev/ruth.git
cd ruth
./setup_env.sh
```

Il setup prepara `.venv`, installa le dipendenze e verifica CUDA. Se e solo se
l'utente non dispone gia' di credenziali Hugging Face e il terminale e'
interattivo, chiede di inserire un **token Hugging Face personale di sola
lettura**. L'inserimento non e' visibile. Crea il tuo token su
https://huggingface.co/settings/tokens .

L'autenticazione e' facoltativa per i modelli pubblici: premi Invio per
saltarla. Un token valido e' salvato da `huggingface_hub` nella cache personale
dell'utente, **non nel repository Git, nel codice o nel Git credential helper**.
Non inserire token in file versionati, screenshot, log o issue.

Chi esegue setup in CI o in una sessione non interattiva non riceve prompt;
puo' usare la variabile d'ambiente `HF_TOKEN` o configurare un secret del
proprio workflow. I secret di GitHub Actions non diventano automaticamente
disponibili sul computer locale.

Per configurare o sostituire il token personale in seguito:

```bash
.venv/bin/python -m model.hf_auth --force
```

Per verificare l'identita' autenticata senza mostrare il token:

```bash
.venv/bin/python -c 'from huggingface_hub import whoami; print(whoami()["name"])'
```

Avvio, dopo l'installazione:

```bash
.venv/bin/python main.py
```


## Calibrazione baseline e contesto GPU-resident

Questa fase diagnostica **non modifica il modello operativo**. Ogni
modello conserva manifest separati in `models/<model-id>/` (ignorati da Git).
Disponibili: Gemma 2 9B IT, Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct.

Per Qwen, prima controllare l'inventario e selezionare la baseline full-GPU:

```bash
.venv/bin/python -m calibration.baseline_discovery \
  --model qwen-2.5-7b-instruct --inventory-only

.venv/bin/python -m calibration.baseline_discovery \
  --model qwen-2.5-7b-instruct --probe-n-ctx 64 --batch 32
```

Dopo che la baseline di un modello e' stata **verificata nativamente**,
la ricerca del contesto usa sempre il GGUF indicato in
`baseline_selection.json`, senza modificare pesi, offload, KV precision
o altri parametri llama.cpp:

```bash
.venv/bin/python -m calibration.gpu_context \
  --model gemma-2-9b-it

.venv/bin/python -m calibration.gpu_context \
  --model llama-3.1-8b-instruct

.venv/bin/python -m calibration.gpu_context \
  --model qwen-2.5-7b-instruct
```

Ogni prova riempie quasi interamente il contesto con un prompt reale e
genera otto token. Verifica dai log che tutti i pesi GGUF e tutta la KV
cache siano sulla GPU, registra i principali buffer compute GPU e i
buffer ausiliari CUDA_Host, e misura il picco NVML. Aumenta il contesto,
poi raffina token per token il limite tra successo ed esaurimento CUDA
confermato. Gli errori ambigui non vengono trattati come OOM salvo
esplicito `--retry-ambiguous-context`. La conferma ripete la misura due
volte. Il risultato e' scritto in `gpu_context_reference.json`, separato
da `baseline_selection.json`.

**ATTENZIONE:** un massimo contesto *diagnostico* appena allocabile non
e' automaticamente stabile o idoneo alla produzione, specialmente con
poco margine VRAM; la selezione del profilo operativo e' una fase
successiva. Nessun upload dei risultati e nessun profiling dei layer
viene effettuato da questi comandi.
