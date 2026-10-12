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

Kaggle usa **un token personale separato** da Hugging Face. Crea il token
su https://www.kaggle.com/settings/api e conserva il valore segreto.
Il nome scelto sul sito (per esempio `ruth-local`) non e' il token.
Il setup, se eseguito da un terminale senza credenziali Kaggle gia'
configurate, chiede il valore con input nascosto. Ruth lo verifica con
Kaggle prima di salvarlo in `~/.kaggle/access_token`, con permessi
`0600` (cartella `~/.kaggle` protetta con `0700`).
La configurazione e' personale, facoltativa e non finisce in Git.

Per aggiungerlo o sostituirlo senza ripetere il setup CUDA:

```bash
.venv/bin/python -m model.kaggle_auth --force
```

Per controllare l'account autenticato senza mostrare il segreto:

```bash
.venv/bin/python -c 'import kagglehub; print(kagglehub.whoami()["username"])'
```

Se usi CI o un terminale non interattivo, configura `KAGGLE_API_TOKEN`
tramite il gestore di segreti dell'ambiente, senza inserirlo nella
riga di comando. Tale variabile ha precedenza sul file personale.
Sono supportate anche le credenziali Kaggle legacy gia' presenti.
Nessun token viene richiesto automaticamente in CI.

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


## Profilazione automatica incrementale

L'interfaccia principale della calibrazione e' ora:

```bash
.venv/bin/python -m calibration.profile_pipeline \
  --model gemma-2-9b-it \
  --source bartowski \
  --engine llama.cpp \
  --method llama_cpp_python
```

Le quattro scelte sono esplicite e opzionali; possono essere estese
quando nuovi fornitori, motori e adapter verranno implementati. Opzioni
non ancora supportate producono un errore chiaro, **mai un fallback
implicito**.

Le caratteristiche hardware sono invece rilevate automaticamente:
backend CUDA dalla GPU NVIDIA, UUID e modello GPU, VRAM totale, RAM,
CPU e versione del runtime. Non esiste un argomento per inventare una
VRAM diversa da quella fisica.

Il comando usa una pipeline riprendibile per fasi:

1. `inventory`: inventario GGUF e priorita' qualitativa.
2. `baseline`: prima quantizzazione interamente caricabile su CUDA.
3. `context_gpu`: massimo n_ctx diagnostico con pesi e KV su CUDA.
4. `mapping`: previsto, non ancora implementato.
5. `runtime`: previsto, non ancora implementato.
6. `context_tuned`: previsto, non ancora implementato.
7. `capacity`: previsto, non ancora implementato.

Ogni candidato e test di contesto viene stampato **durante la prova** con
il risultato e la decisione, senza dover attendere l'output JSON completo.
I report rimangono disponibili in
`models/calibration_runs/<profile_key>/` (`state.json`,
`events.jsonl`, `inventory.json`, `baseline.json`, `context_gpu.json`).
Questa cartella e' esclusa da Git. Il `profile_key` dipende dalle
scelte, dall'hardware misurato e dalla versione del runtime: non si
confondono i profili tra GPU, quantizzazioni o ambienti diversi.

```bash
# Esegui solo il primo passo su tutti i modelli registrati
.venv/bin/python -m calibration.profile_pipeline \
  --model all --until inventory

# Completa solo fino alla baseline, conservando l'inventario
.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct --until baseline

# Riprendi dal primo passo mancante o incompleto
.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct

# Ripeti anche fasi gia' verificate
.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct --rerun

# Mostra anche i passi futuri non ancora implementati
.venv/bin/python -m calibration.profile_pipeline --list-phases
```

I risultati diagnostici non sono un profilo operativo e **non cambiano**
il modello caricato da Ruth. Un run inconclusivo (per esempio per errore
di un worker) rimane `blocked` e potra' essere riprovato. Il processo
non promuove mai un errore inconclusivo a OOM. I comandi manuali delle
singole fasi rimangono disponibili per diagnosi e retrocompatibilita'.


## Fonte Kaggle: Mistral 7B Instruct v0.3

Il quarto modello è `mistral-7b-instruct-v0.3`, con fonte `kaggle`.
Il catalogo iniziale punta alla pagina Kaggle pubblica
`bond005/mistral-7b-instruct-v0.3`. **La pagina di un modello non è
una prova che contenga file GGUF**: Ruth accetta soltanto file .gguf
confermati dal catalogo della variante Kaggle specificata, mai una
sostituzione Hugging Face. La CLI Kaggle 2.2.4 non implementa
`models variations list`, percio' l'handle di una variante reale e' richiesto.

Requisiti: `kaggle` CLI e `kagglehub` Python, installati con
`./setup_env.sh`. Per inserire il valore del token gia' creato su Kaggle
usa `.venv/bin/python -m model.kaggle_auth --force`. In alternativa
la CLI Kaggle supporta OAuth con `kaggle auth login`. Non inserire mai
token negli argomenti, nel repository Git o nei report diagnostici.
I token Hugging Face non autenticano Kaggle.

Prima prova (soltanto metadata, nessun file multi-GB):

```bash
.venv/bin/python -m calibration.profile_pipeline \
  --model mistral-7b-instruct-v0.3 \
  --source kaggle --engine llama.cpp --method llama_cpp_python \
  --kaggle-handle OWNER/MODEL/FRAMEWORK/VARIATION/VERSION \
  --until inventory
```

Sostituisci `OWNER/MODEL/FRAMEWORK/VARIATION/VERSION` con un
handle reale verificato nella pagina Kaggle: e' un segnaposto,
non un handle pronto da eseguire.

Se una pubblicazione Kaggle non espone GGUF, Ruth segnala una fase
`blocked` e non inventa una baseline. È possibile specificare un
handle **reale**, rilevato dalla pagina Kaggle di una variante GGUF:

```bash
.venv/bin/python -m calibration.profile_pipeline \
  --model mistral-7b-instruct-v0.3 --source kaggle \
  --kaggle-handle OWNER/MODEL/FRAMEWORK/VARIATION/VERSION \
  --until inventory
```

Il valore nel comando sopra è solo il formato, **non un handle verificato**.
Puoi specificare anche una pubblicazione di un altro autore su Kaggle:
Ruth ricava `owner/model` dall'handle reale e lo inserisce
nell'identita' del profilo, mantenendo invariato il modello logico
Mistral 7B Instruct v0.3. L'eventuale GGUF deve comunque essere
presente in Kaggle e avere un nome coerente con il modello.
Dopo la conferma dell'inventario, riesegui senza `--until inventory`
per la selezione GPU e il contesto diagnostico.

La preselezione di memoria usa il numero esatto di byte dei tensori
GGUF. Per Kaggle, non avendo al momento lo stesso accesso remoto
parziale all'header usato dal provider Hugging Face, tale verifica
viene effettuata **dopo il download del candidato**. Non vengono
scartati GGUF sulla sola base della dimensione del file, né inventati
margini VRAM. La fonte e l'handle Kaggle sono inclusi nei profili.
