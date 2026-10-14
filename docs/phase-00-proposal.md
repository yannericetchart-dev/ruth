# Fase 00 — Specifica preliminare di comportamento automatico

> **Stato:** revisione 0.3 della proposta, **non approvata**. Requisiti forniti dall'utente inclusi come target e da confermare; **nessun adapter è già funzionante**.
> **Data:** 14 ottobre 2026.
> **Roadmap:** [ROADMAP.md](../ROADMAP.md) (approvata con riserva di modifiche future).
> **Regola:** documento ancora soggetto ad approvazione. Per avviare il lavoro è stata scritta una **bozza limitata** di contratti Python e di registro dichiarativo; nessuna installazione, download, prova hardware o runtime funzionante.

## Obiettivo della fase 00

Definire con precisione cosa significa «completamente automatico» per il motore di profiling e quali condizioni devono essere rispettate prima che il risultato possa essere utilizzato da un altro branch.

**Definizione proposta:** dopo la scelta iniziale di una *policy*, Ruth rileva l'ambiente, prepara un piano, sceglie e scarica modelli autorizzati, prova più candidati, calibra una configurazione, la verifica con inferenza reale e pubblica un profilo utilizzabile. Durante l'esecuzione non chiede conferme impreviste: se un'operazione esce dai limiti autorizzati, registra il motivo e si arresta senza dichiarare successo.

Il progetto procede **in ordine, per fasi sottoposte ad approvazione separata**. L'automazione riguarda il software finale, non le nostre scelte durante lo sviluppo.

## Matrice di requisiti comunicati dall'utente — revisione in attesa di approvazione

I nomi fra parentesi quadre nella richiesta indicano il **primo hardware / codice previsto per sviluppo e verifica sulla macchina dell'utente**, **non** implementazione già avvenuta o approvazione finale. Le altre voci devono essere rappresentate **nel codice come `TO DO`**: possono avere registri, manifest di capability e adapter placeholder, ma non devono essere pubblicizzate come funzionanti o importare obbligatoriamente librerie non installate.

L'obiettivo architetturale è **un'unica API** che permetta a codice client identico di utilizzare GPU NVIDIA, AMD o Intel e CPU Intel/AMD, instradando le differenze nelle implementazioni specifiche. **Non significa dichiarare supporto AMD/Intel funzionante senza test nativi**.

| Dimensione | Da sviluppare/testare per primo | Presenti nel codice come `TO DO` |
| --- | --- | --- |
| Sistema operativo | **GNU/Linux** | Nessun OS aggiuntivo richiesto |
| Architettura CPU | **x86_64 Intel** | **x86_64 AMD** |
| GPU vendor | **NVIDIA** | **AMD, Intel** |
| Backend GPU | **CUDA** | **Vulkan, ROCm, HIP, SYCL, oneAPI, XPU, OpenVINO GPU** |
| Libreria nativa | **GGML CPU (utilizzata da llama.cpp, NON un backend)** | — |
| Tecnologie CPU future | **Nessuna combinazione di inferenza CPU in sviluppo** | **oneDNN, oneMKL, OpenVINO CPU, ONNX Runtime CPU, vLLM CPU, BLIS, OpenBLAS** |
| Motore di inferenza | **llama.cpp** | **vLLM, ONNX Runtime, PyTorch, OpenVINO, MLC LLM, IREE, SGLang** |
| Inizializzazione di llama.cpp | **`llama_cpp_python`** | **`local-server`, `system-library`, `cli`** |

`ONNX Runtime` è il nome tecnico proposto per la voce «onix runtime» indicata dall'utente; la grafia è **da confermare**. Allo stesso modo, `ROCm`/`HIP` e `SYCL`/`oneAPI`/`XPU` possono essere livelli collegati e non necessariamente backend nativi indipendenti: il catalogo deve esplicitare relazioni e identificatori senza inventare compatibilità.

### Livelli e API comuni (proposta, da approvare)

```text
Client futuro (nessuna conoscenza del produttore GPU)
       |
       v
ModelRuntime API comune: load / generate / stream / close
       |
       v
InferenceEngine: llama.cpp [PRIMO], vLLM / ONNX / ... [TO DO]
       |
       v
InitializationAdapter: llama_cpp_python [PRIMO], server/library/cli [TO DO]
       |
       v
ExecutionBackend: CUDA [PRIMO], altri [TO DO]
GGML CPU: libreria nativa usata da llama.cpp, NON un ExecutionBackend
       |
       v
HardwareProvider: GNU/Linux x86_64; Intel CPU/NVIDIA GPU [PRIMI],
                  AMD CPU/AMD GPU/Intel GPU [TO DO]
```

Le classi di interfaccia **non** importano direttamente CUDA, ROCm, SYCL, PyTorch o llama.cpp: il registro seleziona l'adapter disponibile solo quando necessario. Un adapter implementato dichiara compatibilità specifica con motore, modalità di avvio, backend e device; un incrocio non dichiarato valido termina con errore esplicito.

**Contratto minimo per le voci del catalogo** (concettuale, formato definitivo da decidere):

- `id`, `kind` (hardware/backend/engine/initialization), `status` (`TO_DO` / `implemented` / `verified`);
- `supported_os`, `supported_cpu_arch`, vendor/dispositivo compatibile come capacità **dichiarate**, distinte da prove locali;
- combinazioni consentite e dipendenze facoltative;
- `probe()` senza effetti pesanti, `available()` e motivazione d'indisponibilità;
- identificatore dell'adapter e `load()` soltanto quando implementato, con errori tipizzati altrimenti;
- `evidence` di verifica nativa, hardware, versioni e workload prima di marcare `verified`.

Il codice principale chiama sempre le API neutrali. Adapter o provider specifici isolano le differenze NVIDIA/AMD/Intel; nessun ramo specifico del produttore deve comparire nel codice di scelta modello, download, orchestrazione e gestione dei profili. I test mock dell'API possono certificare la **struttura**, non rendere `verified` una combinazione hardware assente.

## Decisioni da prendere adesso (opzioni, non default impliciti)

### D1 — Perimetro hardware / sistema operativo

**Precisazione fornita dall'utente:** GNU/Linux e architettura CPU **x86_64**; CPU Intel come hardware host da rilevare, **GPU NVIDIA + CUDA come solo percorso d'inferenza iniziale**. GGML CPU è una **libreria utilizzata da llama.cpp**, non un backend CPU o una seconda combinazione. CPU AMD e GPU AMD/Intel rimangono `TO DO` come supporto nativo futuro.

**Decisioni aperte:**
- **Vincolo chiarito:** l'unico runtime iniziale da implementare/provare usa NVIDIA + CUDA + llama.cpp + `llama_cpp_python`; nessun fallback CPU in questa fase.
- **Vincolo chiarito:** GGML CPU è una dipendenza/libreria e non può essere passata come valore di `ExecutionBackend`.
- Quanto deve rilevare il provider AMD/Intel in fase 01, prima degli adapter di inferenza: identificazione e diagnosi senza promessa di esecuzione oppure solo voce `TO DO`?

**Nessuna proposta viene assunta come approvata** prima del confronto.

### D2 — Separazione provider / backend / motore / inizializzazione

**Requisiti da confermare:** interfaccia unica e vendor-neutral, con registro esplicito delle combinazioni valide. **Unica combinazione da sviluppare e verificare:** **NVIDIA + CUDA + llama.cpp + `llama_cpp_python`**. La CPU Intel è l'host e GGML CPU è una libreria utilizzata: nessuna delle due aggiunge uno stack runtime CPU.

Altri motori: vLLM, ONNX Runtime, PyTorch, OpenVINO, MLC LLM, IREE, SGLang. Altre modalità per llama.cpp: `local-server`, `system-library`, `cli`. Sono tutte voci di catalogo con stato `TO DO`, senza fallback nascosti.

**Decisioni aperte:** definizione esatta di API e capability registry; gestione versioni degli adapter; rappresentazione di voci che possono essere sia provider esecutivi sia motori (p.es. OpenVINO). Le decisioni specifiche saranno chiuse prima della fase 03.

### D3 — Politica di selezione del modello

- **A: priorità alla qualità entro vincoli di memoria/disco/latency misurati**; velocità come soglia, non obiettivo assoluto.
- **B: priorità alla velocità reale entro una soglia di qualità**.
- **C: punteggio pesato configurabile**, con soglie e pesi esplicitamente approvati.

**Proposta:** A, ma il test di qualità minimo e le soglie vanno definiti nelle fasi 05/08; «modello più grande» non è sinonimo automaticamente di «migliore».

### D4 — Autonomia dei download

- **A: autorizzazione iniziale** che specifica fonti, budget e accettazione licenze; Ruth può scaricare autonomamente entro quei limiti.
- **B: conferma interattiva per ogni download**, più controllo ma automazione incompleta.

**Proposta:** A, con comando `plan` che mostra il costo previsto. Credenziali personali gestite fuori dal repository; i modelli gated non si aggirano.

### D5 — Comportamento in caso di errori

- **A: fallback automatico soltanto entro la policy** (altro candidato, backend o CPU), registrando ogni passaggio, senza nascondere il fallimento.
- **B: stop immediato al primo errore**.

**Proposta:** A per fallimenti recuperabili; fermarsi sui vincoli hard, su errori di sicurezza e su assenza di candidati validi. Niente runtime pubblicato se il test finale non passa.

### D6 — Budget e uso delle risorse

Policy esplicita prima dell'esecuzione: spazio totale scaricabile, spazio libero minimo, massimo numero di candidati, limite di durata/prove, uso delle GPU, riserva di RAM/VRAM, qualità e latenza minime, permesso rete. **Nessun valore numerico è approvato qui**: discuteremo quelli misurabili quando avremo rilevamento hardware e benchmark disponibili. Nel frattempo `plan` deve poter indicare «dato sconosciuto» senza inventarlo.

### D7 — Contratto pubblico di compatibilità (aggiunto)

- **Proposta:** ogni prodotto del profiling pubblica identificativi separati di `hardware_provider`, `execution_backend`, `inference_engine`, `initialization_adapter` e versioni, più prove native.
- **Proposta:** un runtime non cambia vendor, backend o motore da sé; usa soltanto la configurazione già verificata, salvo invocazione esplicita di una nuova procedura di selezione/profiling.
- **Proposta:** capability `TO DO` visibili nelle API informative, ma non disponibili per `load`/`run`; errori espliciti e diagnosticabili.
- **Da approvare:** confini delle quattro interfacce, comportamento dei fallback, semantica di `verified` e firma pubblica per futuri consumer.

## Flusso end-to-end proposto

```text
policy approvata
    ↓
doctor: inventario x86_64 GNU/Linux + provider GPU/CPU + registry di capability (TO DO incluso)
    ↓
plan: combinazioni backend/engine/init verificate + shortlist modelli, costi e vincoli
    ↓
run: download/cache → verifica integrità → baseline nativa
    ↓
benchmark comparabile → massimo contesto operativo
    ↓
KV / parametri / mapping (solo test pertinenti e compatibili)
    ↓
scelta configurazione → validazione isolata indipendente
    ↓
pubblicazione atomica di runtime_profile
    ↓
altro processo: verifica profilo → load → generate / stream → close
```

Tutti i passaggi registrano risultati e motivazioni. Un arresto o crash non cancella i risultati delle fasi completate. `resume` riutilizza soltanto checkpoint ancora compatibili con identità di macchina, modello, backend e workload.

## Interfacce CLI proposte (nomi da approvare)

| Comando | Effetti consentiti |
| --- | --- |
| `doctor` | Inventario di sola lettura; nessun download e nessuna prova pesante |
| `plan` | Valutazione preliminare di candidati, vincoli e costi; nessun download dei pesi |
| `run` | Esecuzione automatica nei limiti di una policy autorizzata e registrata |
| `resume` | Ripresa di prove interrotte con identità ancora compatibile |
| `verify` | Verifica del runtime pubblicato e smoke test reale, nei limiti della policy |

Le opzioni testuali e il formato della configurazione (TOML/JSON) saranno definiti prima dell'implementazione.

## Contratto minimo del risultato (campi concettuali)

Un profilo operativo pubblico deve includere almeno:

- **schema** e versione del contratto;
- **modello**: repository, revisione immutabile, architettura, quantizzazione, file/shard, hash ove disponibile;
- **runtime**: campi distinti per hardware provider, execution backend, inference engine, initialization adapter, loro versioni, parametri e limiti applicati;
- **hardware**: tipo dispositivo, identificatore risolubile e informazioni di compatibilità;
- **risorse**: contesto operativo validato, RAM/VRAM e workload di verifica;
- **evidenza**: risultati del test nativo e del test indipendente, date, stato e motivo della selezione;
- **integrità**: informazioni sufficienti a rilevare un profilo obsoleto;
- **stato**: nessuna promozione senza `verified` effettivo.

**Separazione:** report sperimentali e checkpoint sono privati della pipeline; l'altro branch legge solo il manifest e importa l'API del runtime. In nessun caso deve importare moduli di profiling, chat, voce o visione per generare testo.

## Stati e invarianti

Stati delle singole prove: `pending`, `running`, `verified`, `failed`, `inconclusive`, `skipped`, `unsupported`, `interrupted`.

- La presenza di una GPU **non** certifica CUDA nel backend.
- L'architettura API comune **non** certifica supporto nativo per i vendor non ancora testabili.
- Ogni voce richiesta ma non sviluppabile sulla macchina attuale resta nel **codice** come `TO DO`, selezionabile per ispezione ma non per esecuzione.
- Una modalità llama.cpp (`local-server`, `system-library`, `cli`) non è equivalente all'altra e non si abilita senza test propri.
- Il caricamento di pesi **non** certifica una generazione riuscita.
- La dimensione di un GGUF **non** certifica che entri in VRAM.
- Un OOM va identificato con evidenza, non dedotto genericamente da un crash.
- La migliore configurazione di benchmark **non** è automaticamente stabile in produzione.
- Un fallimento successivo **non** deve sovrascrivere l'ultimo profilo verificato.
- Un'incompatibilità rilevata al caricamento dell'API del runtime produce errore esplicito e non un fallback implicito.

## Requisiti di verifica della fase 00 (prima di qualsiasi codice)

- [ ] L'utente approva esplicitamente la definizione di automazione.
- [ ] L'utente conferma o corregge la matrice completa (CPU, GPU, backend CPU/GPU, inferenza, modalità llama.cpp), incluse sovrapposizioni e nomenclatura.
- [ ] L'utente decide D1–D5 e D7, modificando liberamente le proposte.
- [ ] L'utente accetta che i valori quantitativi di D6 vengano definiti con i dati delle fasi successive, **sempre prima di eseguire prove**.
- [ ] Sono approvate le responsabilità di `doctor`, `plan`, `run`, `resume`, `verify` e del runtime.
- [ ] È accettata la separazione fra dati sperimentali e profilo operativo verificato.
- [ ] È concordato che solo la combinazione realmente testabile può diventare `verified`, mentre tutte le altre restano `TO DO` nel codice.
- [ ] La roadmap resta modificabile; variazioni con impatto sulle scelte già implementate sono documentate e verificate.

## Ciclo di vita dei documenti di progettazione

Questo file è **temporaneo** e serve a discutere la fase 00. Dopo approvazione e implementazione della fase, le decisioni finali entrano nella documentazione stabile e nei contratti di codice; il file temporaneo sarà rimosso con un commit di pulizia, quando l'utente ne confermerà l'inutilità. `ROADMAP.md` rimane come riferimento di sviluppo finché utile. Eliminare un file dal branch **non** lo cancella dalla cronologia Git: eventuali riscritture della storia richiedono consenso esplicito.

### Primo incremento di codice — contratto e catalogo (da approvare)

- `ruth_runtime/contracts.py`: protocolli vendor-neutral e modelli dati di base, ancora soggetti a revisione.
- `ruth_runtime/capabilities.py`: 33 voci, delle quali sei `target` (inclusi CPU Intel **solo host** e libreria GGML CPU) e 27 `TO DO`; nessuna voce `verified`. **Un solo stack iniziale**: NVIDIA/CUDA/llama.cpp/`llama_cpp_python`.
- `ruth_runtime/__init__.py`: superficie di import pubblica preliminare.
- `tests/test_phase00_contracts.py`: test standard library per stati e combinazioni; controllo esplicito che GGML CPU non sia selezionabile come backend.
- Nessun adapter, driver, downloader, profiling o runtime di inferenza è stato implementato. I test eseguiti non costituiscono verifica CUDA.
- **Punto di controllo:** revisione e approvazione del contratto prima della Fase 01.

**Stato finale della fase 00:** non raggiunto; matrice e D1–D7 restano in revisione. Esiste solo una bozza API dichiarativa, da modificare liberamente.
