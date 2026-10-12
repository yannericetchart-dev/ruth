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
Disponibili: Gemma 2 9B IT, Llama 3.1 8B Instruct, Qwen 2.5 7B Instruct e Mistral 7B Instruct v0.3.

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


## Modelli e fonti disponibili

Ruth espone quattro **modelli logici** e sei combinazioni modello/fonte.
La fonte `bartowski` identifica le quantizzazioni GGUF pubblicate su
**Hugging Face**. Le fonti disponibili sono:

| Modello | Hugging Face (`bartowski`) | Kaggle (`kaggle`) |
| --- | --- | --- |
| `gemma-2-9b-it` | `bartowski/gemma-2-9b-it-GGUF` | `google/gemma-2/gguf/2.0-9b-it` |
| `llama-3.1-8b-instruct` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Non disponibile |
| `qwen-2.5-7b-instruct` | `bartowski/Qwen2.5-7B-Instruct-GGUF` | `qwen-lm/qwen2.5/gguf/7b-instruct` |
| `mistral-7b-instruct-v0.3` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | Non disponibile |

`--source` seleziona **una sola fonte per esecuzione**, senza fallback da
Kaggle a Hugging Face o viceversa. Per ciascun modello, Hugging Face e'
la fonte predefinita per mantenere la compatibilita' con i download e
i profili locali preesistenti.

### Inventari senza download

Le due varianti Kaggle sono preimpostate nel catalogo a partire dagli URL
forniti dall'utente: non serve piu' copiare un `--kaggle-handle` per esse.
La presenza di una pagina Kaggle **non conferma ancora i nomi o le
quantizzazioni dei file GGUF**: e' l'inventario remoto a controllarli.

```bash
# Quattro modelli su Hugging Face
.venv/bin/python -m calibration.profile_pipeline \
  --model all --source bartowski --until inventory

# Solo Gemma e Qwen da Kaggle, con i rispettivi handle predefiniti
.venv/bin/python -m calibration.profile_pipeline \
  --model all --source kaggle --until inventory

# Una sola combinazione, ad esempio Gemma su Kaggle
.venv/bin/python -m calibration.profile_pipeline \
  --model gemma-2-9b-it --source kaggle --until inventory
```

Il token Kaggle puo' essere configurato con
`.venv/bin/python -m model.kaggle_auth --force` (input nascosto e cache
personale). Alcuni modelli, incluso Gemma 2, possono richiedere
l'accettazione delle condizioni d'accesso sul sito Kaggle prima della
prima interrogazione.

La CLI Kaggle non offre `models variations list`; Ruth usa direttamente
gli handle di variante registrati nel catalogo. E' possibile sostituire
la variante con `--kaggle-handle OWNER/MODEL/FRAMEWORK/VARIATION[/VERSION]`
soltanto per Gemma o Qwen con `--source kaggle`. L'handle alternativo
viene incluso nell'identita' della calibrazione.

### Diagnosi Kaggle senza download

Se l'inventario Kaggle segnala un errore, Ruth indica una categoria
filtrata (`credentials_missing`, `unauthorized`, `forbidden`,
`not_found`, `cli_syntax`, `network` oppure `unknown`) **senza
stampare la risposta grezza**, che potrebbe contenere segreti.
`not_found` puo' includere risorse nascoste per mancanza di permessi:
non e' una prova che la pagina web non esista. In questo caso Ruth
prova a interrogare `models variations versions list`: se Kaggle
pubblica un numero di versione verificabile, tenta `versions files`
usando l'handle completo. In assenza di versioni confermate non ne
inventa una e mantiene la fase bloccata. Il recupero non scarica
i pesi GGUF.
L'autenticazione a KaggleHub e quella della CLI devono entrambe funzionare
nella stessa sessione utente.

```bash
.venv/bin/kaggle --version

# Testa la CLI autenticata, senza mostrare output potenzialmente sensibile
if .venv/bin/kaggle quota >/dev/null 2>&1; then
  echo "Autenticazione CLI Kaggle: OK"
else
  echo "Autenticazione CLI Kaggle: NON VERIFICATA"
fi

.venv/bin/python -c 'import kagglehub; print(kagglehub.whoami()["username"])'

.venv/bin/kaggle models variations files \
  google/gemma-2/gguf/2.0-9b-it --csv --page-size 5

.venv/bin/kaggle models variations files \
  qwen-lm/qwen2.5/gguf/7b-instruct --csv --page-size 5
```

I comandi `variations files` sono consultazioni di metadati, non
richiedono il download dei pesi. **Non condividere log grezzi che
contengano il token**: rimuovi eventuali token, header e URL firmati.
Per Gemma 2, verifica anche le condizioni di licenza su Kaggle.

Su Qwen Kaggle le quantizzazioni possono essere distribuite in
**shard GGUF** (`...-00001-of-00002.gguf` ecc.). Ruth raggruppa i
file in un singolo candidato logico per quantizzazione, verifica la
presenza di tutti gli shard e ordina le quantizzazioni per priorita'
qualitativa. `FP16` e `Q5_0` sono formati riconosciuti. L'inventario
non scarica pesi: i numeri riportati ora rappresentano **candidati
logici**, non i singoli file.

Per le prove baseline il download Kaggle richiede tutti gli shard
del candidato, a partire dallo shard 1; il lettore GGUF somma i tensori
di tutti gli shard e la verifica nativa CUDA deve coprirli tutti.
Un gruppo mancante/incompleto rimane non classificato e **blocca**
l'avanzamento. La prima prova di una quantizzazione Kaggle potrebbe
richiedere il download di molti gigabyte: non avviare `--until baseline`
senza controllare spazio disco e piano di selezione. Nessuna valutazione
della fattibilita' VRAM viene inferita soltanto dalle dimensioni dei file.

Se il nome GGUF ottenuto da Kaggle non consente di classificare la
quantizzazione in modo affidabile, il candidato resta nell'inventario
come **non classificato** e la pipeline non seleziona una baseline di
qualita' presunta. Una baseline richiede la verifica nativa CUDA di
tutti i pesi e un'inferenza minima riuscita.

Le due fonti mantengono identita' e risultati diversi:
`models/calibration_runs/<profile_key>/` per i report; i pesi Kaggle
sono salvati sotto `models/<model-id>/sources/kaggle/<owner>/<model>/`,
senza sovrascrivere i GGUF Hugging Face gia' presenti in
`models/<model-id>/`. I profili non modificano il modello operativo.
