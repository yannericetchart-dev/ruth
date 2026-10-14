# Ruth — Roadmap del branch `model-profiling`

> **Versione:** 1.4 — roadmap approvata e modificabile. Driver NVIDIA minimo **595.45.04**, baseline CUDA **13.2**; procedura host vuoto con `.venv` e diagnosi dei prerequisiti.
> **Branch:** `model-profiling` (riscrittura da zero sulla base minimale di `main`).
> **Riferimento storico:** `feature/minimal-chat` è consultabile per imparare dalle prove precedenti, **non** è una dipendenza né una base di codice da copiare.
> **Stato attuale:** Fase 01 verificata su hardware reale. Fase 02: diagnosi dipendenze implementata, in attesa di esecuzione sul PC dell'utente. Nessuna inferenza/runtime/profiling operativo.

## 1. Obiettivo e confini

Costruire da zero una procedura **automatica, misurabile, riprendibile e verificabile** che:

1. rilevi hardware, ambiente software e risorse realmente disponibili;
2. determini quali backend di inferenza siano effettivamente utilizzabili;
3. scelga autonomamente un modello compatibile, in base a una politica esplicita;
4. recuperi in modo sicuro i pesi necessari e ne verifichi l'integrità;
5. esegua test nativi reali, misurando capacità, velocità, stabilità e qualità minima;
6. ottimizzi il profilo entro i vincoli fissati, senza falsi successi o assunzioni arbitrarie;
7. produca **un profilo operativo verificato e un'API Python per caricarlo**;
8. permetta a futuri branch (chat, voce, visione, ecc.) di usare il runtime senza rifare il profiling.

**Fuori ambito:** chat, cronologia conversazionale, memoria persistente, voce, visione, interfacce grafiche, agenti, esecuzione di funzioni del modello e deployment remoto. Nessuno di questi componenti deve essere introdotto incidentalmente.

### Regole non negoziabili

- **Una fase per volta:** analisi → opzioni tecniche → decisione dell'utente → implementazione → test → verifica dei risultati → commit.
- **Niente implementazione anticipata:** non si inizia la fase successiva prima della revisione della precedente.
- **Nessuna compatibilità inventata:** un backend compare come disponibile soltanto se implementato e verificato; gli altri restano esplicitamente non supportati.
- **Nessun successo fittizio:** distinguere `non_testato`, `non_supportato`, `fallito`, `interrotto`, `verificato`.
- **Misure prima delle stime:** limiti GGUF, dimensione dei file e VRAM libera sono indizi; non dimostrano che un modello funzioni.
- **Mai confondere profiling e produzione:** un record sperimentale non diventa automaticamente runtime valido.
- **Riproducibilità:** modello, revisioni, file, hardware, backend, versioni, parametri, workload e risultati identificano ogni prova.
- **Sicurezza:** niente token nel repository o nei log; mai cancellare pesi/checkpoint senza consenso; niente upload di dati o modelli.
- **Interventi prevedibili:** policy e limiti di download, spazio disco, GPU/CPU, tempo e tentativi sono dichiarati *prima* dell'esecuzione non supervisionata.
- **Errori trasparenti:** niente fallback nascosti fra modelli, backend, quantizzazioni, GPU e CPU.
- **Modularità:** il runtime deve restare indipendente dall'applicazione che lo consumerà.
- **API universale:** hardware, backend esecutivo, motore di inferenza e modalità di inizializzazione sono assi distinti. Interfacce comuni, implementazioni specifiche isolate per vendor e capacità negoziate esplicitamente: nessun `if nvidia` sparso nell'orchestratore.
- **Solo ciò che si può provare:** implementazione reale soltanto sulle combinazioni verificabili sull'hardware dell'utente. Tutte le altre opzioni richieste figurano **nel registro del codice come `TO DO`**, non come funzionalità funzionanti. Le prove simulate verificano contratti, ma non certificano supporto hardware.
- **Regola dipendenze:** ogni programma, libreria o toolchain necessari deve avere verifica di presenza/versione/capacità e, se mancante, diagnosi o installazione documentata. Nessuna installazione automatica, nessun fallback nascosto; distinguere tool di build opzionali dal runtime.
- **Policy driver/CUDA:** driver NVIDIA **R595 o superiore**, con minimo tecnico **595.45.04** per CUDA **13.2 GA** su GNU/Linux. Richiedere CUDA almeno **13.2** dichiarata dal driver, distinguendola dalla presenza del CUDA Toolkit e dalla prova nativa di inferenza; per stack successivi la versione del binding/toolkit va compatibilizzata e verificata. Riferimento NVIDIA: https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/ (tabella driver per toolkit).
- **Preparazione da host vuoto:** durante lo sviluppo si presume che **Python e tutte le altre dipendenze possano mancare**. La shell standard GNU/Linux avvia `bootstrap.sh --plan` anche senza Python; il comando `--create-venv`, solo su richiesta esplicita, crea `.venv` quando Python >=3.10 e venv sono disponibili. La versione base di Python deve essere installata sull'host (o gestita da un tool di bootstrap futuro) prima di creare il virtualenv.
- **Dipendenze Python SOLO nella `.venv` di Ruth:** non usare pacchetti globali né considerare `pip` un requisito host da controllare. L'installer userà l'interprete `.venv/bin/python` e una selezione di versioni bloccate da approvare, quando sarà implementato. I controlli Python del progetto rifiutano interpreti estranei.
- **Documenti di sviluppo temporanei:** i documenti di proposta sono rimossi da Git dopo la chiusura della relativa fase e il trasferimento delle decisioni stabili nei documenti permanenti, previa verifica con l'utente. Un normale commit di eliminazione **non cancella la cronologia**: riscritture/squash solo con approvazione esplicita.

## 1.1 Matrice preliminare di compatibilità (modificabile, da approvare nella fase 00)

Il progetto è **solo GNU/Linux**, con **architettura CPU x86_64**. Il primo hardware fisicamente testabile è **CPU Intel x86_64 (host) e GPU NVIDIA (inferenza)**. L'inventario CPU non abilita un percorso di inferenza CPU. Per il runtime si svilupperà **una sola combinazione NVIDIA + CUDA + llama.cpp + `llama_cpp_python`**; la disponibilità sarà accertata dai probe, non assunta. AMD/Intel GPU e AMD CPU sono target architetturali **`TO DO`** finché non avremo accesso a test nativi reali.

**Convenzione:** `PRIMO` significa «da implementare e verificare per primo», **non** «già supportato». `TO DO` significa «voce prevista nel codice, senza adapter operativo e con rifiuto esplicito dell'esecuzione». L'utente può cambiare questa matrice prima e durante lo sviluppo.

| Livello | PRIMO (testabile dall'utente) | TO DO (previsto nel codice ma non funzionante) |
| --- | --- | --- |
| OS | GNU/Linux | Nessun altro OS nel perimetro |
| CPU (x86_64) | Intel | AMD |
| GPU | NVIDIA | AMD, Intel |
| Backend GPU | CUDA | Vulkan, ROCm, HIP, SYCL, oneAPI, XPU, OpenVINO GPU |
| Libreria nativa | GGML CPU (dipendenza utilizzata, **non** backend selezionabile) | — |
| Tecnologie CPU future | Nessun backend CPU operativo | oneDNN, oneMKL, OpenVINO CPU, ONNX Runtime CPU, vLLM CPU, BLIS, OpenBLAS |
| Motore di inferenza | llama.cpp | vLLM, ONNX Runtime, PyTorch, OpenVINO, MLC LLM, IREE, SGLang |
| Inizializzazione llama.cpp | `llama_cpp_python` | `local-server`, `system-library`, `cli` |

Per OpenVINO, vLLM, ONNX Runtime, SYCL/oneAPI/XPU e simili i livelli possono sovrapporsi: il codice deve conservare **identità e capacità separate**, evitando di equiparare un motore con un driver GPU, un provider d'esecuzione o una libreria matematica. I nomi, le combinazioni lecite e le reali capacità verranno validati, senza assunzioni di compatibilità fra incroci non testati.

La gerarchia pubblica proposta è `HardwareProvider → ExecutionBackend → InferenceEngine → InitializationAdapter → Runtime`; ogni livello espone un contratto neutrale rispetto al vendor. Gli adapter specifici sono selezionati da un **registro di capacità**, mai tramite import obbligatori di librerie non disponibili. Le voci `TO DO` restituiscono `unsupported/not_implemented` e non effettuano fallback nascosti.

## 2. Contratto di avanzamento

Ogni fase qui sotto è uno **stop obbligatorio**. Prima di iniziarla, concordiamo:

1. **Perché:** obiettivo e requisito osservabile.
2. **Cosa:** interfacce, dati in ingresso/uscita e comportamenti attesi.
3. **Come:** alternative, dipendenze e scelta tecnica motivata.
4. **Quanto costa:** download, disco, memoria, VRAM, tempo, eventuali permessi.
5. **Come si verifica:** test unitari, integrazione ed evidenza nativa richiesta.
6. **Esito:** approvato, da correggere o sospeso.

Il commit della fase contiene **solo** quanto concordato, inclusi test e documentazione. Le modifiche alla roadmap sono sempre ammesse su richiesta dell'utente, anche dopo l'approvazione generale: si documentano motivazione, impatto sulle fasi e versione aggiornata, tramite commit identificabili. Nessuna fase implementativa è approvata per il solo fatto di essere elencata. La riserva di modifiche future è parte dell'approvazione.

## 3. Roadmap dettagliata

### Fase 00 — Specifica del comportamento automatico (DA APPROVARE)
- [ ] Definire l'ambiente vincolante: **GNU/Linux, CPU x86_64**; test iniziali su CPU Intel e GPU NVIDIA, senza presumere le specifiche non ancora misurate.
- [ ] Definire le modalità: ispezione senza effetti (`doctor`), anteprima (`plan`), esecuzione automatica (`run`), ripresa (`resume`), verifica (`verify`).
- [ ] Stabilire priorità della scelta (qualità, latenza, throughput, VRAM, RAM, disco), soglie e budget.
- [ ] Confermare **solo lo stack GPU NVIDIA/CUDA**; il rilevamento CPU Intel x86_64 serve per diagnosi host, non per introdurre inferenza CPU o fallback. GGML CPU va classificata tra le **librerie native** utilizzate.
- [ ] Definire come chiedere una sola volta le autorizzazioni (download/licenze/uso risorse), poi procedere in automatico nei limiti accordati.
- [ ] Specificare il contratto del profilo operativo, stati `TO DO`, interfacce vendor-neutral, matrice delle combinazioni e codici di errore prima di scrivere il motore.
- **Output:** specifica versionata delle policy e degli stati; nessun profiling reale.
- **Gate:** l'utente approva ambito, comandi e comportamento dei fallback.

### Fase 01 — Rilevamento dell'hardware (**inventario verificato**)
- [x] Inventario CPU da `/proc/cpuinfo`: architettura, vendor, modello, flag, thread e core fisici *solo se ricavabili*.
- [x] Inventario RAM da `/proc/meminfo`, usando i byte e lasciando `null` i valori non disponibili.
- [x] Inventario PCI/DRM vendor-neutral per GPU NVIDIA/AMD/Intel; metriche NVIDIA da `nvidia-smi` collegate **solo per corrispondenza PCI**. Telemetria AMD/Intel: `TO DO`.
- [x] Identificazione di più GPU e raccolta della variabile `CUDA_VISIBLE_DEVICES` come contesto, non prova di accessibilità CUDA.
- [x] Controllo dei processi GPU sul PC dell'utente il 14/10/2026 alle 05:56: `nvidia-smi` riporta zero processi, 2 MiB / 6141 MiB di VRAM occupati, driver 580.178.04, CUDA massimo riportato 13.0. **Snapshot**: le fasi di benchmark devono controllare di nuovo grafica/compute e contesa prima di ogni prova.
- [x] Gestione di assenza di GPU, `nvidia-smi` mancante, dati incompleti e piattaforma non supportata senza metriche inventate.
- [x] Test automatici offline per Intel/AMD/NVIDIA, casi con/ senza DRM, parsing CSV e PCI, senza affermare supporto di inferenza.
- [x] **Esecuzione e revisione dell'inventario reale**, riportata dall'utente il 14 ottobre 2026: `21` test standard-library superati; sistema Linux x86_64 compatibile; Intel Core Ultra 7 155H (16 core / 22 thread), circa 15,13 GiB RAM (8,92 GiB disponibili), swap assente; GPU Intel integrata rilevata e GPU NVIDIA RTX 4050 Laptop 6 GiB con driver 580.178.04, VRAM libera circa 5,63 GiB; `warnings: []`. L'identificativo univoco GPU e il report integrale non vengono conservati nel repository.
- **Output:** `python -m ruth_runtime.hardware` stampa un `hardware_report` JSON di sola lettura. Non certifica CUDA, né uno stack LLM.
- **Gate:** hardware e snapshot processi verificati; il monitoraggio dinamico della contesa resta obbligatorio prima dei benchmark. Nessuna conferma di toolkit CUDA o binding LLM da questi dati.

### Fase 02 — Ambiente software e capacità native (**diagnostica realizzata; prova utente da eseguire**)
- [x] Diagnosticare GNU/Linux x86_64, driver NVIDIA e `nvidia-smi`; Python >=3.10 e il pacchetto `llama-cpp-python` vengono letti **esclusivamente dall'interprete `.venv` di Ruth**, mai da site-packages dell'host.
- [x] Applicare il minimo NVIDIA **595.45.04** e verificare il supporto dichiarato CUDA **>=13.2** con `nvidia-smi`; testare versioni minori, mancanti e output invalidi; non confondere con toolkit o inferenza reale.
- [x] Interrogare `llama_supports_gpu_offload()` in un subprocess isolato, distinguendo pacchetto assente, import fallito, binding CPU-only, crash e timeout. GGML CPU è una libreria, non un backend CPU.
- [x] Distinguere presenza pacchetto, GPU offload dichiarato e inferenza reale ancora **non** verificata.
- [ ] Progettare installazione/versionamento riproducibile del backend senza alterazioni silenziose dell'ambiente.
- [x] Diagnostica con `required_now`, `status`, messaggi e `remediation`. **Nessun controllo `pip`**; CMake, compilatore C++ e `nvcc` sono strumenti host opzionali per wheel CUDA valide, necessari soltanto in caso di compilazione da sorgente.
- [x] Progettazione e primo bootstrap shell (`sh bootstrap.sh --plan|--check|--create-venv`) da **macchina senza Python**: diagnosi dinamica di Python e modulo venv, CMake/C++, `nvcc` e `nvidia-smi` con istruzioni per preparare i prerequisiti; nessuna installazione di sistema automatica. Solo `--create-venv` crea il virtualenv su richiesta esplicita.
- [x] Test simulati per Python/pacchetti host assenti e venv mancante, ambiente sbagliato, build CUDA CPU-only e tool host assenti.
- [ ] Definire versioni Python/binding approvate, artefatti, compatibilità driver, installazione riproducibile delle dipendenze **dentro .venv**, rollback e prove su macchina pulita.
- [ ] L'utente esegue `sh bootstrap.sh --plan`, prepara la `.venv` se assente tramite `sh bootstrap.sh --create-venv`, quindi `.venv/bin/python -m ruth_runtime.dependencies`; eventuali installazioni di sistema o pacchetti Python saranno concordate prima.
- [ ] Congelare le versioni e verificare dipendenze native effettivamente caricate durante la successiva fase adapter.
- **Output:** bootstrap POSIX shell anche senza Python; dopo preparazione `.venv`, il comando `.venv/bin/python -m ruth_runtime.dependencies` produce JSON read-only e codice d'uscita 1 per requisiti mancanti; nessun download o installazione da parte del diagnostico.
- **Gate:** versioni e prerequisiti testabili e riproducibili.

### Fase 03 — Backend e adattatore di inferenza
- [ ] Definire API vendor-neutral separate per provider hardware, backend esecutivo, engine d'inferenza e modalità di inizializzazione; capacità, probe, caricamento, tokenizzazione, generazione, streaming, chiusura e metriche.
- [ ] Registrare **tutti** i motori/backend/opzioni di inizializzazione elencati nella matrice §1.1; implementare solo l'incrocio iniziale verificabile e segnare il resto `TO DO` con errori espliciti.
- [ ] Implementare e testare **soltanto** `NVIDIA + CUDA + llama.cpp + llama_cpp_python`; usare GGML CPU come libreria dipendente, non come backend. `local-server`, `system-library`, `cli` e inferenza CPU restano fuori dalla prima implementazione (`TO DO`).
- [ ] Definire backend automatico, vincoli hardware e gestione dei fallimenti.
- [ ] Isolare i processi di prova per errori nativi/CUDA senza contaminare l'orchestratore.
- [ ] Verificare la corretta associazione GPU selezionata ↔ processo.
- **Output:** adapter minimo capace di inferenza reale + report di supporto.
- **Gate:** non basta un modello caricato; deve produrre token realmente.

### Fase 04 — Fonti dei modelli, catalogo e metadati
- [ ] Decidere repository/fonti supportati (Hugging Face, eventuali alternative).
- [ ] Definire identificatori immutabili: modello, revisione, variante, formato, quantizzazione e shard.
- [ ] Leggere metadati e requisiti senza scaricare inutilmente pesi completi quando possibile.
- [ ] Distinguere modelli aperti, modelli con licenza da accettare, modelli gated e credenziali necessarie.
- [ ] Rifiutare formati non leggibili dal backend; distinguere catalogo da compatibilità reale.
- [ ] Documentare i criteri per aggiungere/rimuovere candidati.
- **Output:** catalogo interrogabile, con provenienza e vincoli di licenza.
- **Gate:** nessun download da fonti non autorizzate o con termini non accettati.

### Fase 05 — Selezione automatica dei candidati
- [ ] Tradurre la policy approvata in vincoli *hard* (disco, memoria, backend, formato, licenza) e preferenze *soft* (qualità/velocità).
- [ ] Costruire una shortlist ordinata e motivata, con punteggi separati da dati misurati.
- [ ] Definire un numero massimo di candidati, costi cumulativi e criteri di stop.
- [ ] Specificare cosa accade se tutti i candidati falliscono: errore leggibile, nessun profilo pubblicato.
- [ ] Consentire override manuali espliciti senza disattivare le verifiche.
- **Output:** `selection_plan` visualizzabile senza download o GPU.
- **Gate:** scelta e costi prevedibili prima di usare risorse.

### Fase 06 — Download, cache e integrità
- [ ] Preflight: spazio disponibile, dimensioni previste, permessi, rete e token dove necessari.
- [ ] Download a checkpoint/ripresa, con file temporanei e promozione atomica.
- [ ] Gestione shard, revisioni congelate e layout locale coerente.
- [ ] Verificare hash/dimensioni dove disponibili e coerenza del formato dopo download.
- [ ] Non eliminare né sovrascrivere asset preesistenti senza policy esplicita.
- [ ] Gestire timeout, download parziali, autenticazione scaduta e repository irraggiungibili.
- **Output:** manifest di artefatti locali verificati.
- **Gate:** non avviare prove native su pesi incompleti o corrotti.

### Fase 07 — Baseline nativa per ogni candidato
- [ ] Determinare parametri di partenza e possibili placement CPU/GPU coerenti con backend e budget.
- [ ] Caricare in un worker isolato, registrando esito, tempi, memoria e segnali d'errore.
- [ ] Eseguire prompt concreto e decode reale: non validare soltanto l'allocazione.
- [ ] Verificare ove possibile residenza effettiva di pesi, KV e compute; distinguere evidenza da ipotesi.
- [ ] Gestire ripetizioni, seed e confronto fra run a condizioni comparabili.
- [ ] Conservare fallimenti, OOM espliciti, crash e risultati inconclusivi separatamente.
- **Output:** benchmark di baseline con evidenza nativa e log controllati.
- **Gate:** nessun candidato è «funzionante» senza inferenza completa.

### Fase 08 — Benchmark operativo comparabile
- [ ] Fissare un corpus riproducibile: prefill brevi/lunghi, decode e italiano.
- [ ] Misurare prefill tok/s, decode tok/s, latenza, picchi VRAM/RAM e errori.
- [ ] Eseguire più ripetizioni e rendere visibile la variabilità.
- [ ] Definire una verifica minima della validità delle risposte, senza confonderla con qualità generale.
- [ ] Confrontare solo prove a workload e hardware equivalenti.
- **Output:** tabella dei candidati *verificati* e relativa evidenza.
- **Gate:** criteri misurabili, nessuna classifica costruita su stime mascherate da dati.

### Fase 09 — Calibrazione della capacità di contesto
- [ ] Separare contesto nominale del GGUF, contesto allocabile e contesto operativo sicuro.
- [ ] Ricercare n_ctx con prove complete fino al numero previsto di token generati.
- [ ] Verificare limiti architetturali, presenza GPU, KV e margine di memoria.
- [ ] Gestire bisezione/raffinamento, checkpoint e conferme ripetute.
- [ ] Classificare SIGABRT, OOM e fallimenti ambigui senza attribuire cause non provate.
- [ ] Definire una riserva operativa, da approvare e misurare sulla macchina.
- **Output:** massimo diagnosticato + contesto operativo *separato* e motivato.
- **Gate:** un massimo instabile non viene selezionato come n_ctx del runtime.

### Fase 10 — Ottimizzazione dei parametri di inferenza
- [ ] Fissare una matrice di prova approvata: batch/ubatch, thread, GPU layers, Flash Attention, formati K/V e parametri pertinenti.
- [ ] Escludere preventivamente combinazioni incompatibili secondo il backend realmente installato.
- [ ] Misurare a contesto uguale e con verifiche native della configurazione applicata.
- [ ] Misurare guadagni e regressioni reali, con ripetizioni e arresto su budget.
- [ ] Conservare la baseline come confronto immutabile.
- **Output:** configurazioni candidate e confronto quantitativo.
- **Gate:** si promuove soltanto una configurazione effettivamente verificata.

### Fase 11 — Offload/mapping fine, **solo se giustificato dai dati**
- [ ] Stabilire se il mapping granulare è necessario per il target; diversamente documentare il motivo per cui si salta.
- [ ] Censire tensori, gruppi omologhi, dimensioni, tied weights e vincoli del formato.
- [ ] Verificare che il backend installato consenta placement CPU/GPU realmente controllabile.
- [ ] Testare gruppi e, se necessario, singoli tensori con worker isolati.
- [ ] Confrontare stesso modello, contesto, KV e workload; includere throughput e picchi RAM/VRAM.
- [ ] Salvare checkpoint riprendibili ed evitare decisioni basate su risparmi teorici.
- [ ] Se il mapping è sperimentale/non sicuro, non renderlo requisito del runtime.
- **Output:** miglior mapping riproducibile oppure decisione documentata «non necessario/non supportato».
- **Gate:** nessun monkey patch o hack nativo non verificato entra nel runtime operativo.

### Fase 12 — Scelta finale e qualificazione del profilo
- [ ] Applicare i vincoli hard e la politica di scelta con metriche realmente misurate.
- [ ] Scegliere modello, revisione, file, backend, device, n_ctx, KV, parametri e mapping.
- [ ] Validare più avvii da zero e generazioni successive; misurare stabilità e risorse.
- [ ] Registrare il motivo della scelta e le alternative scartate.
- [ ] Conservare una baseline sicura se l'ottimizzazione sperimentale non migliora il risultato.
- **Output:** `qualified_candidate` completo ma ancora non pubblicato.
- **Gate:** profilo idoneo solo dopo test indipendenti, senza riutilizzare lo stato del worker di ricerca.

### Fase 13 — Manifest e pubblicazione atomica
- [ ] Definire formato versionato del `runtime_profile` con identità completa di file, modello, backend, ambiente e hardware.
- [ ] Salvare evidenze della validazione, parametri operativi e impronta del modello.
- [ ] Pubblicare con scrittura atomica soltanto dopo il gate della fase 12.
- [ ] Preservare l'ultimo profilo funzionante quando una nuova ricerca fallisce.
- [ ] Definire stati `valid`, `stale`, `invalid` in seguito a cambi del file, driver o ambiente.
- [ ] Gestire migrazioni future dello schema senza interpretazioni silenziose.
- **Output:** manifest operativo verificabile, separato dai log e dai report sperimentali.
- **Gate:** un'altra applicazione può identificare esattamente ciò che verrà caricato.

### Fase 14 — API runtime riutilizzabile
- [ ] Definire API Python stabile per lettura/validazione profilo, `load`, `generate`, `stream`, `close`.
- [ ] Non introdurre cronologia chat, prompt di sistema obbligatori né politiche applicative; il client usa la stessa API su device diversi, tramite gli adapter disponibili.
- [ ] Verificare al caricamento modello, adapter, device, versioni e risorse; fallire con diagnosi esplicita se non compatibili.
- [ ] Supportare i messaggi/prompt nel formato consentito dal backend selezionato, con tokenizzazione coerente.
- [ ] Gestire lifecycle, interruzioni, rilascio delle risorse e massimo contesto del profilo.
- [ ] Definire la compatibilità semantica dell'API per branch consumatori successivi.
- **Output:** pacchetto importabile, esempio CLI minimale di smoke test (non una chat).
- **Gate:** un processo Python separato usa il profilo e genera una risposta reale.

### Fase 15 — Qualità, regressioni e automazione end-to-end
- [ ] Test unitari CPU-only per rilevamento, regole di scelta, storage, errori, resumability e profili.
- [ ] Test con adapter simulato per la pipeline completa e per il dispatch vendor-neutral NVIDIA/AMD/Intel; i risultati simulati non promuovono voci `TO DO`.
- [ ] Test di integrazione reali sulla macchina di destinazione (GPU/CPU secondo policy).
- [ ] Test di download interrotto, crash worker, memoria insufficiente, versione cambiata, pesi rimossi e risorse concorrenti.
- [ ] Provare run pulito, run ripreso, run ripetuto e riuso del profilo da altro processo.
- [ ] Prevedere CI per verifiche hardware-independent; verifiche GPU come suite separata.
- [ ] Documentare tempi, limiti, costi e casi in cui l'automazione deve fermarsi in modo sicuro.
- **Output:** prova end-to-end riproducibile e report di accettazione.
- **Gate:** tutti i requisiti minimi verificati sull'hardware target.

### Fase 16 — Documentazione e consegna agli altri branch
- [ ] README essenziale di installazione, `doctor`, `plan`, `run`, `resume`, `verify`.
- [ ] Guida per aggiungere nuovi backend, modelli e fonti.
- [ ] Documentare manifest, API pubblica, stati di errore e policy di fallback.
- [ ] Preparare esempio d'integrazione in un progetto Python esterno senza dipendenze dalla chat.
- [ ] Congelare interfaccia v1 e definire regole per modifiche incompatibili.
- [ ] Verificare che tutti i file runtime necessari siano tracciati/risolvibili e che i segreti e i modelli non vengano commitati.
- **Output:** runtime v1 riutilizzabile e documentazione di handoff.
- **Gate:** approvazione finale dell'utente prima di integrare o fondere il branch.

## 4. Architettura concettuale (NON ancora approvata come struttura di file)

```text
policy/config
      |
      v
hardware + environment discovery
      |
      v
backend capability registry
      |
      v
model catalog -> candidate selector -> download verifier
      |                                   |
      +-----------------------------------+
      |
      v
isolated native workers -> measured evidence / resumable journal
      |
      v
context / KV / mapping experiments
      |
      v
candidate qualification -> verified runtime profile (atomic)
                                     |
                                     v
                             reusable runtime API
                                     |
                                     v
                       future branches (out of scope)
```

Principio: **l'orchestratore del profiling e il runtime operativo sono moduli distinti**. I consumatori futuri importano l'API runtime, non la pipeline di calibrazione. L'infrastruttura sperimentale può usare worker e log complessi; il runtime deve restare minimale e affidabile.

## 5. Interfacce previste (solo proposta, non implementate)

| Funzione | Proposta di comportamento |
| --- | --- |
| `doctor` | Rileva capacità, senza scaricare o usare CUDA per benchmark |
| `plan` | Elenca scelte, rischi e budget prima di toccare i modelli |
| `run` | Esegue dall'inizio in automatico entro la policy approvata |
| `resume` | Riprende checkpoint compatibili senza ripetere le prove già valide |
| `verify` | Controlla manifest e avvia uno smoke test in processo separato |
| API Python | Carica un profilo validato ed espone generazione/streaming/close |

**Da decidere:** nomi definitivi dei comandi, formato della configurazione (TOML/JSON/YAML), installazione delle dipendenze, API esatta, ubicazione degli artefatti. Questa tabella non autorizza ancora scelte implementative.

## 6. Condizioni di accettazione dell'intero branch

Il progetto è completato soltanto se:

- [ ] Da installazione supportata, un singolo comando `run` completa il flusso senza decisioni interattive **dopo** aver ricevuto una policy/autorizzazione iniziale valida.
- [ ] `plan` rende trasparenti selezione, download previsto, carico hardware e limiti.
- [ ] Un modello è considerato valido soltanto dopo un test nativo di prompt e generazione.
- [ ] Profilazione e runtime mantengono identità, configurazione e tracciabilità coerenti.
- [ ] L'algoritmo non inventa backend, compatibilità, metriche o cause dei crash.
- [ ] Interruzioni e riavvii preservano i checkpoint e l'ultimo profilo operativo verificato.
- [ ] Il runtime può essere importato e utilizzato senza moduli di chat/calibrazione.
- [ ] Test unitari + smoke test end-to-end superati sull'ambiente target.
- [ ] Nessun dato sensibile o modello di grandi dimensioni è stato aggiunto a Git.
- [ ] La documentazione spiega esattamente come riprodurre il risultato.

## 7. Decisioni aperte da discutere PRIMA del relativo codice

1. **Priorità del sistema:** qualità massima, latenza, velocità, consumo o compromesso? Con quali limiti?
2. **GPU/CPU:** quali vendor supportare subito? È consentito fallback CPU? Come gestire GPU impegnata?
3. **Backend iniziale:** quale motore supportare davvero nella v1 e quali candidati rimandare?
4. **Catalogo:** quali famiglie/parametri/quantizzazioni/sorgenti includere? Come gestire licenze?
5. **Internet e download:** limite GB, spazio minimo, autorizzazioni, cache e modalità offline?
6. **Budget sperimentale:** durata massima, massimo contesto, tentativi, limite memoria, quando interrompere?
7. **Misure:** corpus, soglie di affidabilità, qualità minima e regola di comparazione.
8. **Profilo operativo:** posizione su disco, schema, semantica di scadenza e retrocompatibilità.
9. **Interfaccia Python:** firma pubblica vendor-neutral e formato di input/output; selezione di hardware, backend, motore e inizializzazione con capacità dichiarate.
10. **Pubblicazione:** definizione di «pronto», release, test in CI e criteri di merge.

Queste non sono domande da risolvere tutte subito: ciascuna viene affrontata **prima della fase che la richiede**, presentando opzioni e conseguenze.

## 7.1 Gestione della documentazione di sviluppo e pulizia Git

- `ROADMAP.md` rimane documento di riferimento finché serve a governare l'intero sviluppo.
- `docs/phase-XX-proposal.md` è **temporaneo**: alla chiusura di ogni fase, trasferire contratti approvati, decisioni, test e risultati nei file permanenti strettamente necessari; poi proporre/rimuovere il documento temporaneo con commit di pulizia.
- Evitare la proliferazione di file di appunti nel branch: nessuna prova locale, log o report voluminoso entra in Git per default.
- Prima della consegna v1: audit dei file tracciati, rimozione dei documenti temporanei esauriti e verifica che il runtime non dipenda dai documenti.
- **Cronologia Git:** un commit che cancella file li rimuove dal checkout corrente ma **non li elimina dai commit precedenti**. Qualsiasi squash, rebase o riscrittura dello storico richiede consenso specifico, perché modifica commit condivisi.

## 8. Registro di avanzamento

| Fase | Stato | Nota |
| --- | --- | --- |
| Roadmap | **Approvata, modificabile** | Approvazione generale del 14 ottobre 2026; revisioni future consentite |
| 00 | **Bozza di contratto da revisionare** | Interfacce e catalogo dichiarativo nel codice, test CPU-only; nessun adapter operativo |
| 01 | **Conclusa (inventario statico)** | 21 test dell'utente, hardware reale e snapshot processi NVIDIA verificati; recheck dinamico richiesto prima dei benchmark |
| 02 | **Bootstrap/diagnostica da verificare** | Python/venv isolati, preflight driver NVIDIA >=595.45.04 e CUDA driver >=13.2; installazione versionata ancora da definire |
| 03–16 | **Non iniziate** | Nessun runtime o profiling operativo ancora sviluppato |

**Prossima azione:** dopo il riavvio del driver, eseguire `sh bootstrap.sh --plan`, test e `.venv/bin/python -m ruth_runtime.dependencies` e confermare driver >=595.45.04, CUDA supportata >=13.2; se manca una dipendenza obbligatoria, preparare le istruzioni di installazione specifiche. Non installare pacchetti o driver automaticamente. Inferenza ancora non verificata.
