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

Le quattro scelte sono esplicite e opzionali. La CLI accetta
`--engine llama.cpp|exllamav2|vllm|aphrodite|tensorrt-llm` come
**opzioni censite**, ma solo `llama.cpp` con `llama_cpp_python`
e' oggi eseguibile. Gli altri motori vengono segnalati come **da fare**
prima di qualunque prova GPU, senza conversioni di formato, installazioni
automatiche o fallback impliciti.

```bash
.venv/bin/python -m calibration.profile_pipeline --list-engines
```

La matrice di compatibilita' registra, per ciascun motore, se sono
previsti spostamenti CPU dei pesi per layer, offload KV e i formati
dichiarati nella tabella fornita dall'utente. E' **metadato di
pianificazione**, non una certificazione runtime di tutte le versioni
o di ogni funzionalita'.

Le caratteristiche hardware sono invece rilevate automaticamente:
backend CUDA dalla GPU NVIDIA, UUID e modello GPU, VRAM totale, RAM,
CPU e versione del runtime. Non esiste un argomento per inventare una
VRAM diversa da quella fisica.

Il comando usa una pipeline riprendibile per fasi:

1. `inventory`: inventario GGUF e priorita' qualitativa.
2. `baseline`: prima quantizzazione interamente caricabile su CUDA.
3. `layer_matrix`: matrice per blocco e tensore **in sola lettura**, nessun offload.
4. `context_gpu`: massimo n_ctx diagnostico con pesi e KV su CUDA.
5. `mapping`: vero placement layer/tensori, **da fare**.
6. `runtime`: ottimizzazione parametri llama.cpp, **da fare**.
7. `context_tuned`: contesto con mapping ottimizzato, **da fare**.
8. `capacity`: contesto esteso con RAM di sistema, **da fare**.

Ogni candidato e test di contesto viene stampato **durante la prova** con
il risultato e la decisione, senza dover attendere l'output JSON completo.
I report rimangono disponibili in
`models/calibration_runs/<profile_key>/` (`state.json`,
`events.jsonl`, `inventory.json`, `baseline.json`,
`layer_matrix.json`, `context_gpu.json`).
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


## Matrice dei layer senza CPU offload

Dopo una baseline full-CUDA gia' riuscita, si puo' analizzare la
distribuzione dei pesi **senza fare una nuova prova GPU**:

```bash
.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct \
  --source bartowski \
  --until layer_matrix
```

La pipeline recupera `inventory` e `baseline` gia' verificate e crea
`models/calibration_runs/<profile_key>/layer_matrix.json`.
Se esiste un report della versione precedente, Ruth ricrea **soltanto**
`layer_matrix` con schema 2, senza ritentare la baseline CUDA.

Ogni voce in `tensor_index` contiene nome GGUF esatto, quantizzazione
del tensore, forma, byte e MiB, categoria e blocco, `tensor_role`,
selettore `exact_cpu_override_regex`, collegamenti con altri tensori
dello stesso sottosistema, indici dei blocchi adiacenti e vincoli
speciali (`placement_constraint`). Ogni blocco contiene inoltre
`same_block_tensor_names`, `module_type`, `previous_block`,
`next_block`. I rapporti di vicinanza e sottosistema sono **deduzioni
dai nomi GGUF**, non connessioni del grafo computazionale dimostrate.
La collocazione CPU indicata e' una **candidatura da provare**, mai una
garanzia: `tensor_cpu_native_verified_count=0`.

In Gemma 2, `token_embd.weight` puo' essere condiviso da embedding e
proiezione d'uscita e non e' spostabile per uso in modo indipendente.
In Llama, `rope_freqs.weight` puo' avere piu' istanze native da
verificare separatamente. I vincoli sono esplicitati anche in
`placement_constraints`.

### Inventari dettagliati per tutti e quattro i modelli Hugging Face

Per verificare **nomi, dimensioni, quantizzazione dei singoli tensori
e struttura di tutti i blocchi** anche senza una baseline full-GPU:

```bash
.venv/bin/python -m calibration.layer_matrix \
  --model all --source bartowski
```

Il comando usa l'elenco dei file e letture *ranged* dei **soli header**
dei quattro GGUF di riferimento `Q4_K_M` dal catalogo. I report finiscono
in `models/calibration_runs/remote_hf_matrices/<model-id>/<file>.json`.
Ogni rapporto contiene `provenance=remote_gguf_header_only_no_runtime_verification`
e `baseline_native_verified_all_gpu=false`: **non** prova che il
modello sia caricabile integralmente su CUDA. Non scarica i pesi
GGUF e non sposta layer; richiede l'accesso a Hugging Face.

La matrice locale Qwen `Q5_K_L` e' invece ricavata dal GGUF della
baseline gia' verificata e ha
`provenance=local_cuda_baseline_verified`. I due file non si
sovrascrivono. Le analisi con quantizzazioni differenti sono distinte
anche se la topologia dei blocchi fosse identica.
Per Qwen Q5_K_L questo report contiene:

- tutti i tensori con nome, tipo GGUF, forma e byte serializzati;
- una riga per ciascun blocco `blk.N`, con dimensioni e gruppi
  `attention`, `feed_forward`, `normalization`, ecc.;
- tensori fuori dai blocchi (embedding, output, normalizzazioni);
- scenari **puramente teorici** per futuri prefissi consecutivi di
  blocchi CPU da 1/2/4/8/16/... layer (ove presenti);
- marcatori `cpu_offload_executed=false` e
  `operational_profile_changed=false`.

Le cifre sono **byte GGUF serializzati**, non risparmio VRAM previsto.
Non includono KV cache, scratch/compute buffer, overhead CUDA, costi
di trasferimento CPU/GPU o latenza. La matrice non abilita
`mapping` e non sposta alcun layer in RAM. Se un modello multi-shard
e' incompleto, la fase viene bloccata.

### Motori futuri: capacita' per tipologia di profilo

La seguente tabella replica la classificazione condivisa per
pianificare gli adapter, senza considerarla una verifica empirica
delle release correnti:

| Motore (`--engine`) | CPU offload pesi per layer | KV host/offload | Formati previsti | Stato |
| --- | --- | --- | --- | --- |
| `llama.cpp` | Si', totale/parziale | No, nella matrice di pianificazione | GGUF | Implementato |
| `exllamav2` | No | No | EXL2, GPTQ | Da fare |
| `vllm` | No (eccezione CPU-only fuori scope) | Si', swapping asincrono | AWQ, GPTQ, FP16/BF16 (no EXL2) | Da fare |
| `aphrodite` | No | Si', configurabile | EXL2, GPTQ, AWQ | Da fare |
| `tensorrt-llm` | No | Si', host pinned | Engine TensorRT compilati | Da fare |

Solo la matrice dei layer GGUF con `llama.cpp` e' implementata.
Le altre strategie di profilo richiederanno sorgenti, formati, worker,
misure e verifiche dedicate: Ruth non tenta di eseguirle.

## Modelli e fonti disponibili

Ruth espone quattro **modelli logici** e cinque combinazioni modello/fonte.
La fonte `bartowski` identifica GGUF pubblicati su **Hugging Face**.
L'unico modello supportato anche su Kaggle e' Qwen 2.5 7B Instruct:

| Modello | Hugging Face (`bartowski`) | Kaggle (`kaggle`) |
| --- | --- | --- |
| `gemma-2-9b-it` | `bartowski/gemma-2-9b-it-GGUF` | Non supportato |
| `llama-3.1-8b-instruct` | `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` | Non supportato |
| `qwen-2.5-7b-instruct` | `bartowski/Qwen2.5-7B-Instruct-GGUF` | `qwen-lm/qwen2.5/gguf/7b-instruct` |
| `mistral-7b-instruct-v0.3` | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | Non supportato |

Gemma 2 e' stata rimossa dalle fonti Kaggle intenzionalmente: il
GGUF presente nella variazione verificata e' descritto dal publisher
come **pesi non quantizzati**, da quantizzare separatamente con
`llama-quantize`. Non e' un'alternativa utile al catalogo GGUF
gia' quantizzato scelto per la calibrazione quality-first di Ruth.
L'inventario Gemma su Hugging Face rimane valido e invariato.

`--source` seleziona una sola fonte, senza fallback implicito.
`--model gemma-2-9b-it --source kaggle` viene rifiutato: non si
interroga piu' il modello Kaggle. La fonte di default per tutti
i quattro modelli e' `bartowski`.

### Inventari senza download

```bash
# Quattro modelli su Hugging Face
.venv/bin/python -m calibration.profile_pipeline \
  --model all --source bartowski --until inventory

# Solo Qwen 2.5 7B Instruct su Kaggle
.venv/bin/python -m calibration.profile_pipeline \
  --model all --source kaggle --until inventory

# Equivalente, per il singolo modello Kaggle
.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct --source kaggle --until inventory
```

Gli inventari gia' completati vengono riutilizzati senza nuove
interrogazioni; i report precedenti restano per diagnostica, ma i
vecchi profili Gemma/Kaggle **non sono selezionabili dal catalogo**.

### Diagnosi Kaggle senza download

Il token puo' essere configurato in modo sicuro con
`.venv/bin/python -m model.kaggle_auth --force`, senza metterlo
nel repository. Ruth filtra gli errori della CLI (`license_required`,
`credentials_missing`, `unauthorized`, `forbidden`, `not_found`,
`cli_syntax`, `network`, `unknown`) senza riportare stderr grezzo.
Stampa `LICENZA DA ACCETTARE` solo se l'API segnala esplicitamente
che manca il consenso; un `404` da solo non prova un problema di licenza.

```bash
.venv/bin/kaggle --version

if .venv/bin/kaggle quota >/dev/null 2>&1; then
  echo "Autenticazione CLI Kaggle: OK"
else
  echo "Autenticazione CLI Kaggle: NON VERIFICATA"
fi

.venv/bin/python -c 'import kagglehub; print(kagglehub.whoami()["username"])'

.venv/bin/kaggle models variations files \
  qwen-lm/qwen2.5/gguf/7b-instruct --csv --page-size 5
```

Non condividere token, header di autenticazione o URL firmati.
La CLI non offre `models variations list`: Ruth usa la variazione
esplicita dal catalogo. Per Qwen e' disponibile anche
`--kaggle-handle OWNER/MODEL/FRAMEWORK/VARIATION[/VERSION]`, con
identita' di calibrazione dedicata alla variante scelta.

### GGUF suddivisi in shard e baseline

Su Qwen Kaggle, piu' file
`...-00001-of-00002.gguf`, `...-00002-of-00002.gguf` ecc.
costituiscono un unico candidato logico. Ruth verifica che tutti
gli shard siano presenti, ne aggrega i tensori per la verifica
nativa CUDA e blocca i gruppi incompleti. L'inventario considera
i **candidati logici**, non il numero di file fisici.

L'inventario non scarica pesi. La fase `baseline` invece puo'
scaricare molti gigabyte, specialmente con GGUF Kaggle multi-shard;
su Kaggle non sono disponibili le stesse verifiche remote dei pesi
usate per Hugging Face. Non avviare `--until baseline` su Qwen
Kaggle senza prima verificare spazio disco e strategia di preflight.
In ogni caso il codice non promuove automaticamente la baseline a
modello operativo e richiede prova full-CUDA senza CPU spill.

I report diagnostici di ogni fonte restano separati sotto
`models/calibration_runs/<profile_key>/`. I GGUF Kaggle sono copiati
sotto `models/<model-id>/sources/kaggle/<owner>/<model>/`,
senza sovrascrivere i GGUF Hugging Face nella directory del modello.
