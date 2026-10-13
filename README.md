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
5. `mapping`: **gruppi omologhi di tensori per ruolo**, screening,
   benchmark full-GPU e prove CPU/GPU ripetute, **implementate**.
   Mapping operativo ancora da verificare nelle fasi successive.
6. `runtime`: ottimizzazione parametri llama.cpp, **da fare**.
7. `context_tuned`: contesto con mapping ottimizzato, **da fare**.
8. `capacity`: contesto esteso con RAM di sistema, **da fare**.

Ogni candidato e test di contesto viene stampato **durante la prova** con
il risultato e la decisione, senza dover attendere l'output JSON completo.
I report rimangono disponibili in
`models/calibration_runs/<profile_key>/` (`state.json`,
`events.jsonl`, `inventory.json`, `baseline.json`,
`layer_matrix.json`, `context_gpu.json`, `mapping.json`).
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


## Fase 2: classifica dei gruppi tensori e benchmark contro baseline

**La fase 2 non sposta interi blocchi**, perche' la fase 1 ha gia'
dimostrato che i modelli selezionati sono completamente residenti
in VRAM. Si cerca invece di liberare VRAM per buffer e futuro n_ctx
senza ridurre materialmente le prestazioni di inferenza.

1. **Gruppi omologhi**. Da GGUF e `layer_matrix` Ruth raggruppa tutti
   i tensori di tipo `blk.*.<suffix>`, ad esempio
   `blk.*.attn_k.weight`, `blk.*.attn_norm.weight`,
   `blk.*.ffn_down.weight`. Nessuna lista hardcoded delle famiglie.
   Conserva nome, forma, quantizzazione, numero di blocchi, byte
   GGUF **unici** e numero di materializzazioni native previste.
   Non somma due volte embedding/output condivisi di Gemma 2,
   `rope_freqs.weight` riutilizzato nativamente o altri nomi
   fuori blocco. Questi ultimi sono censiti separatamente,
   ma non vengono automaticamente spostati.

2. **Screening computazionale, senza esclusioni arbitrarie**.
   Le dimensioni GGUF uniche e un proxy delle operazioni per token
   producono una classifica e documentano il costo relativo.
   **Tutti i gruppi `blk.*.<suffix>` sono testati**, inclusi quelli
   minori di 16 MiB o con proxy computazionale incompleto.
   Non e' previsto un numero massimo di gruppi. Il filtro
   non esclude semplicemente perche' il vantaggio stimato sia
   piccolo: saranno le misure native a stabilirlo.
   I pesi non appartenenti a blocchi (per esempio embedding/output
   condivisi e rope materializzata più volte) sono censiti
   separatamente e non vengono spostati automaticamente: servono
   prove dedicate dei vincoli di condivisione.

3. **Contesto realmente esteso oltre la fase 1**. Dal GGUF
   si recupera il limite architetturale e si sceglie il primo
   valore di contesto potenza di due **strettamente superiore**
   al `confirmed_max_gpu_resident_n_ctx` della fase 1, senza
   superare il limite del modello. Esempio: Gemma con massimo
   full-GPU 256 -> test `n_ctx=512` con **383 token di prompt,
   128 di generazione** (un token riservato). Qwen Kaggle con
   massimo 7424 -> `n_ctx=8192`, prompt 8063 e 128 decode.
   Queste prove non possono essere sostituite dal vecchio
   benchmark sotto il limite di VRAM.

4. **Baseline host-KV**. Tre processi isolati caricano **tutti i
   pesi GGUF su GPU**, come nella fase 1, ma richiedono
   `offload_kqv=False`, cioe' **cache KV su RAM CPU durante
   l'inferenza**. Il placement KV host deve risultare nei log
   nativi; non ci si affida a un fallback automatico.
   La baseline deve eseguire l'intero prompt esteso e 128 token
   di decode, misurando tempo, throughput, VRAM totale,
   margine VRAM minimo campionato e RSS del processo.

5. **Gruppi di tensori con KV GPU**. Uno alla volta, Ruth
   sposta soltanto i tensori omologhi del gruppo sulla CPU,
   mantenendo gli altri pesi full-CUDA e richiedendo
   `offload_kqv=True`. Le prove native devono confermare
   la KV interamente su CUDA e il compute principale su GPU.
   Si provano **tutti** i gruppi `blk.*`, con tre processi
   ciascuno, stesso `n_ctx`, 128 decode, prompt, batch e
   quantizzazione della baseline. Errori di singoli gruppi
   sono registrati senza interrompere la ricerca.

6. **Confronto a contesto identico**. Il parametro principale e'
   il cambiamento di velocita' rispetto alla baseline
   *pesi CUDA + KV CPU*. Se almeno uno tra prefill e decode
   migliora di oltre il 3%, l'altro non peggiora di oltre
   il 3% e la dispersione resta limitata, il gruppo viene
   considerato promettente. Un gruppo senza guadagno misurabile
   non viene promosso. La differenza di VRAM fra i due casi e'
   registrata come **delta**, insieme al margine VRAM
   realmente disponibile: il candidato puo' usare **piu'
   VRAM** della baseline perche' ospita la KV su GPU, e questo
   non e' di per se' un fallimento. I MiB GGUF del gruppo
   non sono un risparmio VRAM nativo automaticamente garantito.

Il report `mapping.json` (schema 4) contiene gruppo, ranking,
baseline host-KV, prove con KV GPU, throughput, picchi VRAM/RAM e
verdetti. Questo **non e' un profilo operativo** e non modifica
i risultati della fase 1. I report schema 1, 2 e 3 (prove
precedenti a contesto inferiore) **non sono riutilizzabili**
per il nuovo benchmark: cambiano sia il contesto sia la residenza KV.
Restano disponibili come diagnosi e Ruth conserva il vecchio
`mapping.json` in `mapping.prior_schema3.json` (oppure
`mapping.prior_schema2.json` per i report schema 2).
Dalla nuova fase 2 sono riprendibili solo checkpoint schema 4
con prova nativa e workload strettamente identici.
Un gruppo senza KV GPU valida viene registrato come fallito;
la baseline host-KV fallita blocca il confronto complessivo.
Le fasi `runtime`, `context_tuned` e `capacity` rimangono
non implementate.

```bash
cd ~/Dev/ruth
git pull --ff-only
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Partire da Qwen Kaggle, riutilizzando tutte le prove fase 1:
.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct --source kaggle --until mapping

# Dopo un report valido, estendere alle quattro fonti Bartowski:
.venv/bin/python -m calibration.profile_pipeline \
  --model all --source bartowski --until mapping
```

## Fase 1 completa: audit delle cinque combinazioni modello/fonte

Le cinque configurazioni sono **quattro modelli Hugging Face** e
**Qwen 2.5 7B Instruct Kaggle**. La fase 1, per ciascuna fonte, e'
considerata completa **soltanto** quando sono verificati tutti:

1. inventario completo;
2. scelta della migliore quantizzazione che ha superato la prova CUDA
   nativa della baseline, con inferenza minima riuscita;
3. peso GGUF **scaricato e presente su disco** (tutti gli shard nel caso Kaggle);
4. \`context_gpu\`: massimo \`n_ctx\` diagnostico confermato piu' volte,
   con tutti i tensori del modello su CUDA, KV su CUDA e compute principale
   su CUDA. Sono possibili piccoli buffer ausiliari \`CUDA_Host\`, registrati
   esplicitamente e non equivalenti a KV in RAM.

I quattro report da \`--remote-header\` dimostrano unicamente la struttura
GGUF remota. **Non sono un risultato della fase 1** e non garantiscono
download, qualita' selezionata, caricamento CUDA ne' contesto residente.

Audit non invasivo di quanto gia' misurato sulla macchina:

\`\`\`bash
.venv/bin/python -m calibration.phase1_audit --model all --source all
\`\`\`

Il report consolidato e' salvato come
\`models/calibration_runs/phase1_audit.json\`, con cinque righe,
quantizzazione, GGUF locale, prova nativa full-CUDA e \`n_ctx\`
confermato. L'audit non effettua download o nuovi probe CUDA,
non cambia profili operativi e **termina con codice 1** se manca
anche una sola configurazione.

Per completare le fasi non ancora misurate su Hugging Face:

\`\`\`bash
.venv/bin/python -m calibration.profile_pipeline \\
  --model all --source bartowski --until context_gpu
\`\`\`

La pipeline **riutilizza** inventari/baseline/matrici esistenti:
non usare \`--rerun\` salvo una richiesta intenzionale di rifare
tutte le prove. La prova di contesto avvia worker CUDA isolati e
puo' richiedere numerose iterazioni/tempo. Il massimo riportato
e' diagnostico: non implica un margine operativo sicuro.

Quando un worker di contesto non produce JSON valido, Ruth registra
`worker_returncode`, `worker_signal` e lo stato del protocollo.
**Un SIGABRT scarta quel n_ctx senza un secondo tentativo**;
la bisezione prosegue verso contesti inferiori. Per la calibrazione
si distingue ora il **massimo empirico** dal motivo del fallimento:
se la ricerca raggiunge due valori adiacenti `N` funzionante
e `N+1` fallito con SIGABRT o fallimento nativo di risorse,
e `N` supera le due conferme full-CUDA, `N` e' il massimo
empirico per il runtime e i parametri fissati. La ricerca e'
completa e `confirmed_max_gpu_resident_n_ctx=N`, con
`maximality_proven=true` e
`maximality_basis=empirical_fixed_runtime_search`.
Questo **non e' una dimostrazione matematica** della capacita'
fisica della VRAM o della causa di ogni crash. Un SIGABRT generico
rimane `failure_cause_unresolved=true`, mai OOM inventato.
Per errori ambigui che non sono SIGABRT e non hanno evidenza
CUDA esplicita rimane invece la prudenza: tali casi non
certificano automaticamente il massimo.

I report diagnostici delle vecchie ricerche gia' concluse vengono
riconciliati **senza eseguire nuovi probe GPU** quando i log salvati
provano il contesto inferiore full-CUDA, le conferme native,
il crash SIGABRT al token immediatamente successivo e la
coerenza con la baseline. La pipeline conserva una copia
`context_gpu.pre_empirical_reconciliation.json` del report
originale, aggiorna il journal e permette all'audit di mostrare
`VALIDATO (massimo empirico)` e la fase 1 completata.
Eseguire la pipeline fino a `context_gpu` senza `--rerun`
per effettuare questa riconciliazione; l'audit resta di sola lettura.
Nessuna quantizzazione o profilo operativo viene cambiato.

Se un worker segnala esplicitamente
`CUDA error: the resource allocation failed` durante
`cublasCreate_v2`, Ruth conserva la categoria
`native_cuda_resource_allocation_failure`, distinta da un OOM CUDA.
La ricerca di contesto verifica pesi, KV e compute principale in GPU;
piccoli buffer ausiliari CUDA_Host non equivalgono a KV CPU.
Il massimo diagnostico appena allocabile non e' necessariamente
sicuro nell'uso operativo: occorre margine VRAM sufficiente.

Su Qwen Hugging Face Q5_K_L, un test a \`n_ctx=2048\` ha misurato
6139,3 MiB su 6141 MiB: appena 1,7 MiB di margine campionato,
non una riserva operativa sicura. Ruth non modifica automaticamente
batch, KV, mapping o quantizzazione per far risultare la prova valida.

## Kaggle: somma degli shard e controllo VRAM

### Nomi delle quantizzazioni e ordine di prova

Il codice distingue la convenzione dei nomi dalla qualita' misurata.
In `llama-quantize`, i nomi brevi `Q3_K`, `Q4_K` e `Q5_K`
sono alias rispettivamente di `Q3_K_M`, `Q4_K_M` e `Q5_K_M`.
Quindi `Q5_K` viene ordinato **nella stessa fascia di
`Q5_K_M`**, non come `Q5_K_L`. Se entrambi sono presenti,
a parita' di fascia si prova prima il nome canonico `Q5_K_M`
e poi l'alias `Q5_K`: e' un criterio deterministico di
ordinamento, non una superiorita' misurata. Le versioni `_L` sono
ricette miste diverse; non esiste la regola generale
`Q?_K == Q?_K_L`. L'alias si applica alla classificazione
nominale, **non** dimostra che due GGUF prodotti da editori
diversi siano identici o abbiano lo stesso punteggio di qualita'.

L'ordine delle famiglie e' una priorita' per scegliere quale
modello verificare prima per la residenza CUDA, non un risultato
di accuratezza/perplessita' misurato. Una verifica a posteriori
della qualita' richiede prove riproducibili sullo stesso dataset
(per esempio una valutazione perplexity), non soltanto il nome
del file, la dimensione o il numero di token del contesto.
I dati locali di `layer_matrix` consentono di ispezionare
le codifiche reali dei singoli tensori, senza attribuire
automaticamente loro un ranking di accuratezza.

Kaggle segue la stessa ricerca quality-first gia' usata per
Hugging Face: considera le quantizzazioni nell'ordine di qualita',
scarta quelle troppo grandi e verifica nativamente su CUDA le altre.

L'unica differenza nel controllo preventivo e' che un GGUF Kaggle
puo' essere suddiviso in piu' shard. Ruth raggruppa tutti i file
della quantizzazione e usa la somma delle dimensioni fornite dal
catalogo:

```text
dimensione_gguf = somma(dimensioni_di_tutti_gli_shard)
se dimensione_gguf > VRAM_totale:
    scarta questa quantizzazione senza scaricarla
altrimenti:
    scarica gli shard mancanti e verifica tutti i pesi su CUDA
```

La regola non contiene nomi o numeri hardcoded di Qwen, non richiede
il `config.json` del modello e non impone budget di download.
Prima di scaricare Ruth verifica soltanto che sul disco ci sia
spazio per la cache KaggleHub e la copia locale; riutilizza gli
shard presenti e non elimina automaticamente modelli precedenti.

Esempio con RTX 4050 (6141 MiB): FP16 e Q8_0 Qwen vengono
esclusi dalla somma dei file (rispettivamente circa 14532 e
7723 MiB). Q6_K (~5964 MiB) supera il filtro preventivo ma,
come verificato dal test nativo dell'utente, fallisce per OOM CUDA.
Il successivo Q5_K_M (~5193 MiB) puo' quindi essere provato
senza alcun parametro aggiuntivo.

```bash
cd ~/Dev/ruth
.venv/bin/python -m calibration.kaggle_preflight \
  --model qwen-2.5-7b-instruct

.venv/bin/python -m calibration.profile_pipeline \
  --model qwen-2.5-7b-instruct --source kaggle --until baseline
```

Il confronto usa le dimensioni **dei file GGUF**, non i byte dei
soli tensori. Inoltre alcune misure del catalogo Kaggle possono
essere arrotondate: e' un filtro preventivo pratico, non una
prova matematica d'impossibilita' vicino alla soglia. Hugging
Face continua a usare la lettura dell'header GGUF remoto per
verificare i pesi senza scaricare il modello. Per ogni candidata
Kaggle non esclusa, la prova nativa CUDA resta necessaria.
Se il catalogo non restituisce una dimensione attendibile, Ruth
si ferma e non salta una candidata ignota.

Il preflight Kaggle non scarica pesi e non avvia prove GPU.
La fonte Kaggle mantiene una baseline indipendente da Bartowski.

Dopo aver completato le pipeline, rieseguire \`phase1_audit\`.
Solo cinque righe \`COMPLETA\` permettono di dichiarare
verificata l'intera fase 1.

### Matrici locali dopo la baseline

La CLI \`calibration.layer_matrix\` ora lavora **di default sui GGUF
gia' scaricati e validati dalla baseline**, senza avviare baseline
mancanti:

\`\`\`bash
.venv/bin/python -m calibration.layer_matrix \\
  --model all --source all
\`\`\`

Se alcune baseline sono assenti, quelle configurazioni sono
segnalate come non verificate e il comando esce con stato nonzero.
Ogni report locale viene salvato in
\`models/calibration_runs/<profile_key>/local_matrix_analysis/\` per
non sovrascrivere il report della pipeline.

La vecchia lettura dei soli header Hugging Face e' opzionale:

\`\`\`bash
.venv/bin/python -m calibration.layer_matrix \\
  --model all --source bartowski --remote-header
\`\`\`

Il file remoto porta \`baseline_native_verified_all_gpu=false\` e
\`current_baseline_location=remote_header_only_not_verified\` per
non confondere la struttura con l'evidenza CUDA.

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

Sono esportati anche quattro CSV affiancati al JSON:
`layer_matrix_tensors.csv`, `layer_matrix_blocks.csv`,
`layer_matrix_links.csv` e `layer_matrix_constraints.csv`.
Sono tabelle direttamente consultabili/filtrabili e non contengono
mapping eseguibili.

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
  --model all --source bartowski --remote-header
```

Il comando usa l'elenco dei file e letture *ranged* dei **soli header**
dei quattro GGUF di riferimento `Q4_K_M` dal catalogo. I report finiscono
in `models/calibration_runs/remote_hf_matrices/<model-id>/<file>.json`.
Ogni rapporto contiene `provenance=remote_gguf_header_only_no_runtime_verification`
e `baseline_native_verified_all_gpu=false`: **non** prova che il
modello sia caricabile integralmente su CUDA. Nelle quattro directory
sono presenti anche i CSV della matrice con nomi identici, ma file
e modello di riferimento differenti. Non scarica i pesi
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


### Esperimento Fase 1: matrice completa KV e Flash Attention

La ricerca comprende le **12 combinazioni ammissibili** di K e V in
`f16`, `q8_0`, `q4_0`: tre con Flash Attention OFF (V=f16)
e nove con Flash Attention ON. La configurazione FA OFF/K=f16/V=f16
è la baseline storica `n_ctx=257`: è riportata, ma non rieseguita.

```bash
python -m calibration.kv_profiles_phase1 \
  --model gemma-2-9b-it \
  --baseline-ctx 257 \
  --tokens 16 --batch 32 --confirm-rounds 2 \
  --output phase1_kv_profiles_complete.json
```

La sequenza di test è:

| Flash Attention | K | V | Note |
|---|---|---|---|
| OFF | f16 | f16 | baseline già misurata |
| OFF | q8_0 | f16 | |
| OFF | q4_0 | f16 | |
| ON | f16 | f16 | |
| ON | f16 | q8_0 | |
| ON | q8_0 | f16 | |
| ON | q8_0 | q8_0 | |
| ON | f16 | q4_0 | |
| ON | q4_0 | f16 | |
| ON | q8_0 | q4_0 | |
| ON | q4_0 | q8_0 | |
| ON | q4_0 | q4_0 | |

Ogni configurazione parte da `n_ctx=258`. Un fallimento iniziale
interrompe solo quella configurazione; la causa dell'errore resta
registrata. I limiti reali derivano dal metadato GGUF
`<arch>.context_length`: la ricerca e i worker rifiutano
contesti maggiori (Gemma-2: **8192**).

Quando un profilo **confermato** raggiunge esattamente il limite del
modello, vengono saltati i profili ancora non misurati che usano lo
**stesso stato Flash Attention** e precisioni K e V entrambe uguali
o inferiori, con almeno una strettamente inferiore. Ad esempio, se
ON/q8_0/q8_0 arriva a 8192, saltiamo ON/q8_0/q4_0,
ON/q4_0/q8_0 e ON/q4_0/q4_0. ON/f16/q4_0 non è
dominato, quindi resta da testare. Le prestazioni non sono deducibili
dalla sola dominanza di precisione: questo filtro serve a evitare
test superflui **di capacità di contesto**, non a scegliere il profilo
più rapido.

Il report è salvato dopo ogni profilo e contiene il motivo del salto
(`skipped_dominated_at_model_limit`) e il profilo dominante.
Il parametro `--max-ctx` può restringere la ricerca, ma non
aumentare il limite del GGUF: raggiungere una soglia `--max-ctx`
inferiore al limite **non** giustifica alcun salto.

Il nuovo report ha schema v3 ed è separato dai precedenti. I risultati
già confermati vengono riutilizzati con parametri identici; usare
`--rerun` per ricominciare. La baseline originale e i profili
operativi rimangono invariati. Codice d'uscita 0 indica profili
confermati, baseline riutilizzata o salti per dominanza; un profilo
fallito o inconclusivo produce uscita diversa da zero.


### Benchmark comparativo Fase 1: throughput, qualità e punteggio

Dopo `phase1_kv_profiles_complete.json`, confrontare soltanto i profili
la cui inferenza è già stata confermata in GPU a ciascun contesto:
512 (9 candidati), 768 (3), 1280 (1) per il report Gemma-2 attuale.
Il repository include ora `calibration/corpora/kv_eval_it_v1.txt`,
un campione UTF-8 originale, **sintetico ma scritto in linguaggio naturale**,
con 12 sezioni tematiche e circa 3300 parole. Non contiene frasi duplicate
artificialmente per riempire il contesto ed è versionato per ripetibilità.
Il worker verifica in modo rigoroso che il tokenizer del modello produca
almeno `n_ctx` token. È un corpus esplorativo, non un benchmark di qualità
rappresentativo della distribuzione reale delle richieste: inoltre questa
implementazione valuta sempre il prefisso iniziale del corpus fino a `n_ctx`,
quindi non tutte le 12 sezioni partecipano a ogni contesto.

```bash
python -m calibration.kv_benchmark \
  --phase1-report phase1_kv_profiles_complete.json \
  --corpus calibration/corpora/kv_eval_it_v1.txt \
  --contexts 512,768,1280 \
  --repetitions 3 --decode-tokens 16 --quality-points 64 \
  --output kv_quality_speed_benchmarks.json
```

Per la velocità ogni configurazione viene provata in **tre worker CUDA
separati**, con tutti i pesi e KV in GPU e verifica nativa di placement,
precisione e Flash Attention; salvati prefill tok/s, decode tok/s, picco
VRAM, margine e mediana. `prompt + generazione = n_ctx`, con
`decode_tokens=16`, per tutti i profili del medesimo contesto.

Per la qualità il confronto è tra K/V FP16 e K/V quantizzate con
**stesso stato Flash Attention, pesi GPU e KV entrambe in RAM**:
questo mantiene costante la collocazione KV, altrimenti FP16
non sarebbe interamente GPU ai contesti richiesti. Il benchmark
calcola **PPL campionata** (64 posizioni next-token, teacher forcing
sullo stesso corpus) e KL(P_FP16 || P_quant) media sull'intero
vocabolario a quelle posizioni, senza buffer logits `n_ctx × vocab`.
Le qualità qui misurate isolano meglio la quantizzazione **a KV CPU**,
ma non certificano l'errore numerico della KV GPU; sono separate dalle
misure di throughput GPU. Non confondere la PPL campionata con la
perplexity classica su tutti i token del corpus.

Per ogni `n_ctx`, il report calcola un **punteggio diagnostico 0–100**
soltanto dove velocità e qualità sono state validate, con pesi
espliciti: KL (30%), PPL relativa (20%), decode (25%), prefill
(15%), margine VRAM campionato (10%). Il punteggio dipende
dalla scala KL scelta (`--kl-scale`, default 0,01 nat) e dalla
soglia di margine (`--headroom-target-mib`, default 64).
Le velocità sono normalizzate al migliore candidato **dello stesso**
contesto: non confrontare punteggi tra lunghezze diverse.

Se la baseline qualità CPU FP16, il candidato o un probe di velocità
falliscono, il punteggio rimane `null`, non viene inventato.
Il flag `headroom_target_met` non prova che il profilo sia
operativamente sicuro. Il benchmark non cambia la baseline
originale né promuove un profilo in produzione.


### Passi di allocazione KV e recupero contesto tramite CPU (Fase 2)

Dopo aver completato `phase1_kv_profiles_complete.json` e
`kv_quality_speed_benchmarks.json`, individuare prima i salti
effettivi dei buffer KV dichiarati dal runtime nativo:

```bash
python -m calibration.kv_memory_steps \
  --phase1-report phase1_kv_profiles_complete.json \
  --boundaries 256,512,1280 \
  --output phase1_kv_memory_steps.json
```

Per ogni profilo Flash/K/V Ruth prova contesti ai bordi del blocco
(`b-1`, `b`, `b+1`) in processi isolati. Il controllo usa **pesi GPU
e KV CPU** per osservare i buffer anche quando la KV GPU non entra in
VRAM; salva la dimensione KV nativa e l'eventuale salto in MiB.
Se la vecchia Fase 1 dispone già di entrambi i valori GPU validati ai
confini, registra anche l'incremento effettivo del buffer KV GPU.
Il delta KV CPU è etichettato `host_proxy`: **non** rappresenta una
misura diretta della VRAM necessaria per un carico GPU completo.
Una differenza nulla tra `n_ctx=257` e `512` è plausibile, perché
entrambi possono ricadere nello stesso blocco KV; non prova che 512
sia realizzabile senza altro spazio. La dimensione del blocco non viene
assunta come dato di fatto senza riscontro nei log.

Ricercare poi combinazioni di **gruppi di tensori omologhi** da
spostare in CPU per recuperare `n_ctx=512` a **KV GPU**:

```bash
python -m calibration.kv_step_offload \
  --phase1-report phase1_kv_profiles_complete.json \
  --memory-steps-report phase1_kv_memory_steps.json \
  --benchmark-report kv_quality_speed_benchmarks.json \
  --corpus calibration/corpora/kv_eval_it_v1.txt \
  --reference-ctx 257 --target-ctx 512 \
  --max-groups 3 --max-candidates 20 --max-validated 3 \
  --repetitions 3 --tokens 16 --quality-points 64 \
  --output phase2_kv_step_offload_512.json
```

Sono candidati i profili storici sotto il target oppure quelli che
non hanno superato il probe `258` per un fallimento esplicito delle
risorse CUDA. Un fallimento a 258 **non** certifica un massimo pari a
256: l'algoritmo riprova il riferimento a 257.

La selezione delle combinazioni è **limitata** da
`--max-groups`/`--max-candidates`, quindi non esplora
necessariamente tutte le sottofamiglie possibili. Le dimensioni GGUF
e il delta della KV sono proxy per scegliere l'ordine di prova.
Il risparmio VRAM effettivo viene invece misurato mediante NVML al
medesimo `n_ctx=257`, confrontando tutti i pesi GPU con i gruppi
offloadati. Ogni candidato deve poi completare **tre inferenze
indipendenti** a 512, con placement nativo verificato per ciascun
tensore e KV nativa in GPU, prima di ricevere lo stato confermato.

Le configurazioni validate ripetono il benchmark prefill/decode
GPU-resident e la misura qualitativa PPL campionata + KL con la
**stessa KV CPU** nel riferimento e nel candidato. Il report salva
`host_group_ids`, `host_tensor_names`, contesto, prestazioni,
qualità, VRAM, verifiche e un punteggio diagnostico ricalcolato
insieme ai profili full-GPU a 512 già misurati.

Non si modifica nessun profilo operativo. I risultati possono
includere `CUDA_Host` (pinned host): la collocazione del tensore
sul lato host non dimostra che il calcolo venga eseguito dalla CPU;
l'effetto prestazionale è misurato e non presunto.


### Ricerca adattiva: gruppi, blocchi, singoli tensori e bundle appresi

Il nuovo esperimento `calibration.kv_adaptive_tensor_offload` parte
**esclusivamente dalla baseline storica FA OFF / K=FP16 / V=FP16,
n_ctx=257**. Riutilizza le misure di `kv_memory_steps` per annotare
il passo osservato, senza usare il proxy in RAM come prova del
risparmio effettivo di VRAM.

```bash
python -m calibration.kv_adaptive_tensor_offload \
  --phase1-report phase1_kv_profiles_complete.json \
  --memory-steps-report phase1_kv_memory_steps.json \
  --targets 512 \
  --scouting-repeats 2 --confirm-repeats 3 \
  --max-tests 250 --max-frozen 40 --max-backtracks 2 \
  --output phase2_adaptive_tensor_groups.json
```

Per esplorare più tappe in una sola ricerca, usare
`--targets 512,768,1024`, sempre entro il limite architetturale GGUF.
La lista dei target è **esplicita**: il passo non viene assunto
universale come 256 e il report KV deve essere consultato prima di
scegliere i target.

**Algoritmo sperimentale:**

1. Misura la baseline con tutti i pesi e KV in GPU a `n_ctx=257`.
2. Confronta a **pari n_ctx** tutte le famiglie omologhe e tutti i
   blocchi completi (resto della configurazione fissato).
3. Seleziona il gruppo con il migliore risparmio NVML rispetto alle
   perdite misurate di prefill e decode.
4. Dentro al gruppo selezionato prova ogni **singolo tensore** in
   isolamento rispetto ai tensori già congelati. Memorizza
   `ΔVRAM`, `Δprefill`, `Δdecode` e tempo combinato
   prompt+generazione. Sposta provvisoriamente il migliore.
5. Dopo ogni scelta tenta il target, ripetendolo **tre volte in worker
   distinti** prima di confermarlo. Se è ancora OOM continua
   costruendo il bundle. I sottoinsiemi sono validati dai log nativi
   tensore per tensore: la KV deve restare in GPU.
6. Se nessun tensore del gruppo libera VRAM misurabile singolarmente,
   ma il **gruppo intero** sì, permette l'offloading del gruppo come
   bundle atomico: la memoria allocata per blocchi può mascherare
   l'effetto delle singole rimozioni.
7. Se un ramo resta senza mosse utili, può annullare fino a
   `--max-backtracks` scelte; se raggiunge il target salva un
   `learned_group` contenente tutti i nomi fisici dei tensori
   spostati e prosegue dal nuovo n_ctx confermato.
8. Salva le misure a ogni prova e riutilizza il checkpoint con
   parametri identici. `--rerun` ricomincia.

**Anchor di profiling (schema greedy v3).** Se l'anchor
storico `n_ctx=257` con FA OFF / K=V=FP16 fallisce per una causa di
risorse CUDA, Ruth controlla se il report KV contiene una **misura
nativa reale** del salto `256 → 257`. Soltanto quando quel salto
esiste e corrisponde a memoria KV aggiuntiva, tenta automaticamente
`n_ctx=256` con tutti i pesi e KV in GPU, verificandolo due volte.
Se 256 non viene confermato, si arresta. Se viene confermato,
profilazione di gruppi e singoli tensori avvengono a **256**; il target
**512** resta invariato e richiede tre conferme indipendenti.
Il report registra `historical_baseline_ctx=257`,
`actual_profiling_start_ctx=256`, il fallimento iniziale, le due
conferme, e il proxy CPU-KV `256 → 512` (quando misurabile).
Non equivale a promuovere un nuovo massimo Fase 1 né garantisce che
256 sia stabile in ogni condizione.

L'anchor vuoto usa esattamente le regole native originali all-GPU.
Il nuovo schema invalida il checkpoint v2: usare un file diverso,
per esempio `phase2_adaptive_tensor_groups_v3.json`, o `--rerun`.
I diagnostici conservano i record nativi KV e le misure NVML.
Se anche 256 fallisce, confrontare il worker originale
`calibration.gpu_context`, la VRAM già occupata prima del caricamento
e gli errori nativi. Un OOM con margine NVML campionato positivo può
essere causato da un'allocazione successiva non riuscita.

Il ranking è **greedy**, dunque non garantisce la migliore soluzione
globale. `--max-tests` limita il numero di configurazioni distinte;
ogni configurazione può avere più ripetizioni.

**Riprendere dopo aver esaurito il budget**: mantenere lo stesso
`--output`, gli stessi parametri di misurazione e aumentare soltanto
`--max-tests` (esempio `250 → 500`). Ruth riutilizza le configurazioni
già salvate in `measured_configs`, ricostruisce le selezioni greedy
dalle misure persistenti e avvia nuovi worker CUDA soltanto per le
configurazioni mancanti. Non serve `--rerun`: quella opzione ricomincia
e sovrascrive il checkpoint. Non diminuire `--max-tests` e non modificare
profili, contesti o altri parametri, altrimenti la ripresa viene bloccata.

```bash
python -m calibration.kv_adaptive_tensor_offload \\
  --phase1-report phase1_kv_profiles_complete.json \\
  --memory-steps-report phase1_kv_memory_steps.json \\
  --benchmark-report kv_quality_speed_benchmarks.json \\
  --corpus calibration/corpora/kv_eval_it_v1.txt \\
  --targets 512 --max-tests 500 \\
  --output phase2_adaptive_tensor_groups_v3.json
```
 I valori NVML
sono misure campionate del picco del processo, non dimensioni
di singoli buffer né certificazioni di stabilità operativa.

È possibile confrontare i bundle finalisti anche per PPL/KL campionata
e punteggio contro le misure full-GPU già ottenute a pari contesto:

```bash
python -m calibration.kv_adaptive_tensor_offload \
  --phase1-report phase1_kv_profiles_complete.json \
  --memory-steps-report phase1_kv_memory_steps.json \
  --benchmark-report kv_quality_speed_benchmarks.json \
  --corpus calibration/corpora/kv_eval_it_v1.txt \
  --targets 512,768 \
  --output phase2_adaptive_tensor_groups_512_768.json
```

La qualità è misurata con KV CPU **sia nel riferimento che nel
candidato**, mentre il test di velocità e capacità usa KV GPU;
questi due risultati restano distinti. I benchmark tra diverse
impostazioni FA/KV sono confronti multiobiettivo esplorativi,
non misure isolate della sola quantizzazione. Il profilo operativo,
la baseline e la Fase 1 storica non vengono modificati.

Il file contiene `scanned_families`, `measured_configs`,
`stages[].group_screenings`, `stages[].individual_screenings`,
`stages[].frozen_tensors`, `learned_groups` e
`final_host_tensor_names`, oltre ai dati dei ripetuti test CUDA.


### Ottimizzare il bundle confermato senza soglie VRAM

Dopo che `phase2_adaptive_tensor_groups_v3.json` ha confermato
`n_ctx=512`, si può cercare una combinazione di tensori più veloce
**senza vincolo di margine VRAM**. Nessun `min_headroom`, né 64 MiB,
né altri MiB imposti. Rimane obbligatorio che ogni configurazione
completi tre inferenze native indipendenti, mantenendo il contesto e
la KV in GPU, con placement per tensore validato.

```bash
python -m calibration.kv_post_optimize \
  --source-report phase2_adaptive_tensor_groups_v3.json \
  --target-ctx 512 \
  --max-candidates 48 --max-rounds 2 --max-tests 160 \
  --decode-weight 0.7 --confirm-repeats 3 \
  --benchmark-report kv_quality_speed_benchmarks.json \
  --corpus calibration/corpora/kv_eval_it_v1.txt \
  --output phase2_post_optimize_512.json
```

Il motore parte dai tensori del `learned_ctx_512` validato e
misura *a parità di n_ctx* la rimozione, la sostituzione e l'aggiunta
di singoli tensori CPU. Il ranking usa esclusivamente le velocità
di **decode (70%) e prefill (30%)** in forma di somma pesata dei
logaritmi dei throughput; modificare `--decode-weight` per preferenze
diverse. La memoria residua NVML è **misurata e riportata**, ma
**non fa parte del punteggio e non stabilisce una soglia minima**.
La verifica di ammissibilità è il successo effettivo del worker
CUDA, non una stima della memoria. Il report include
`best_host_tensor_names`, `measured_configs`, `rounds` e
una `pareto_front` prefill/decode. Il miglior gruppo trovato
può essere diverso da quello iniziale; l'originale non viene
sovrascritto né perso. Se indicati, corpus e benchmark eseguono
un controllo PPL/KL finale quando la ricerca si completa.

Questa ricerca locale è limitata dal budget e non garantisce
l'ottimo globale. Può riprendere incrementando **solo**
`--max-tests`, conservando esattamente lo stesso `--output`
e gli altri parametri. Se si esaurisce il budget, il risultato
`budget_exhausted_best_valid` conserva il miglior gruppo
confermato finora, ma segnala che la ricerca non è completa.
L'opzione `--rerun` ricomincia e scarta il checkpoint di
ottimizzazione (non il report greedy sorgente).
