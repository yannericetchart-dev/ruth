# Fase 02 — Build CUDA nativa: acquisizione automatica dei sorgenti, build ancora da implementare

**Ambito:** GNU/Linux x86_64, Python del progetto in `.venv`, NVIDIA CUDA Toolkit **13.2**, RTX 4050 (`sm_89`). Una build CUDA **non** introduce un backend di inferenza CPU: le librerie GGML CPU possono essere dipendenze native, non una scelta di runtime.

## Stato realmente verificato

Sul PC dell'utente: **73 test** Python superati; `nvcc V13.2.86`, driver NVIDIA `595.99.02`; test nativo `cuda_smoke --run` compilato, eseguito e verificato sulla GPU `0000:01:00.0` (`sm_89`). **Non** sono ancora installati `llama-cpp-python` o i suoi prerequisiti Python. Non sono stati verificati inferenza LLM o benchmark.

## Identità sorgenti (immutabile per questa prova)

| Sorgente | Riferimento |
|---|---|
| Repository Python | https://github.com/abetlen/llama-cpp-python |
| Release di riferimento | `v0.3.36` |
| Git commit sorgente Python | `1652066e0af45f2313b339670ef9555e8a54e545` |
| Submodule `vendor/llama.cpp` | `0c1e57098bba43ac29e6e3b677cdceebdd22334f` |
| Backend CUDA | `-DGGML_CUDA=ON` |
| Architettura | `-DCMAKE_CUDA_ARCHITECTURES=89` |
| Toolkit | `/usr/local/cuda-13.2/bin/nvcc` |
| Generatore | `Unix Makefiles` con `make` disponibile |
| Parallelismo iniziale | 2 processi di compilazione per contenere l'uso della RAM |

La configurazione upstream di `v0.3.36` usa il backend di build `scikit_build_core.build`, richiede `scikit-build-core[pyproject]>=0.9.2` e CMake >=3.21. Questo è un *vincolo minimo*, non una versione già bloccata. L'eventuale wheel ufficiale `cu132` rimane **fallback soltanto manuale**, non la via principale.

## 1. Verifica delle dipendenze senza cambiare nulla

Dalla directory del progetto, dopo `git pull --ff-only`:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m ruth_runtime.install_plan
.venv/bin/python -m ruth_runtime.source_build_plan --plan
```

L'ultimo comando esamina `git`, `cmake --version` (minimo 3.21), `c++`, `make`, stato Toolkit, prerequisiti NVIDIA e, se già presenti, i due commit del checkout. **Nessun clone, pip, download, compilazione o installazione.** Se manca un programma, il JSON indica la procedura di ripristino (intervento sul sistema solo previa approvazione).

Campi attesi in fase iniziale: `build_prerequisites_met=true` se tutti i tool sono disponibili; `source_status=missing` finché non si acquisiscono i sorgenti; `dependency_lock_ready=false`, `ready_to_build=false`, `build_authorized=false`. Il piano **non** usa automaticamente il wheel qualora un prerequisito manchi.

L'**ingresso applicativo principale** è adesso `./ruth`: `./ruth doctor` diagnostica la `.venv`, `./ruth plan` analizza anche la build CUDA e `./ruth setup --allow-downloads` richiama il medesimo acquisitore Git con un'unica autorizzazione al download dei sorgenti. La vecchia interfaccia `source_build_plan --prepare` resta disponibile come comando tecnico. Questo non avvia ancora installazioni Python: vedere `docs/phase-02-venv-runtime.md`.

## 2. Acquisizione automatica dei sorgenti — un solo comando autorizzato

La preparazione dei sorgenti **non richiede operazioni Git manuali**. Il flusso esamina i prerequisiti, scarica il commit fissato di `llama-cpp-python` e il submodule `llama.cpp`, controlla provenienza, commit e albero pulito, poi verifica il checkout finale.

Il comando di Ruth è:

```bash
.venv/bin/python -m ruth_runtime.source_build_plan --prepare
```

**`--prepare` autorizza esplicitamente questo download da GitHub e le scritture nella sola `.ruth-src/`**. Non domanda conferme intermedie, non installa librerie, non cambia il driver, non compila e non esegue fallback. La modalità di default e `--plan` restano read-only.

La procedura è idempotente: se i due commit e l'albero sono già verificati, li riutilizza; se la destinazione è assente, clona in staging privato, effettua checkout pinned e promuove il contenuto solo dopo validazione. Se il checkout esiste ma è alterato, è un symlink o non corrisponde alle revisioni richieste, **blocca senza sostituire**. In caso di errore di rete non seleziona il wheel alternativo; uno staging incompleto viene eliminato. Il clone è limitato al tag `v0.3.36` con `--depth 1`, submodule al commit esatto; nessuna richiesta di credenziali interattive.

Il JSON contiene `acquisition.status` (`acquired_verified`, `already_verified`, `blocked`, `failed`) e `plan_after`. L'esito richiesto per proseguire è `acquisition.source_verified=true` e `plan_after.source_verified=true`, mentre `ready_to_build=false` rimane corretto finché manca il lockfile Python.

In una futura pipeline `run`, la stessa funzione sarà chiamata automaticamente **quando la policy iniziale autorizza i download dei sorgenti**. Non sarà richiesto all'utente di copiare comandi Git in ogni fase. La verifica di download ammessi e limiti generali della pipeline deve ancora essere implementata.

## 3. Python: lockfile e build isolata nel progetto — ancora DA IMPLEMENTARE

Prima di compilare, selezionare **versioni e SHA-256** dei wheel Python compatibili con Python 3.12/Linux x86_64 per:

- build: `scikit-build-core[pyproject]` e **tutte** le sue dipendenze transitive;
- runtime: `typing-extensions`, `numpy`, `diskcache`, `jinja2`, `MarkupSafe` e le eventuali dipendenze transitive addizionali;
- eventuali strumenti di packaging richiesti dal metodo selezionato.

Installazione **solo con** `.venv/bin/python -m pip`, con lock completo e verifica hash. Non usare `pip` dell'host, `--system-site-packages` o build-isolation in grado di recuperare dipendenze non fissate dalla rete. Vietata la sostituzione silenziosa con wheel CPU-only. La policy di sicurezza iniziale vieta `LlamaDiskCache` e la lettura di cache su disco non attendibili; applicarla nell'adapter e coprirla con test.

La ricetta prevista dopo la chiusura del lockfile è una **build locale del wheel Python a partire dai sorgenti verificati**, con:

```text
CUDA_HOME=/usr/local/cuda-13.2
CMAKE_GENERATOR=Unix Makefiles
CMAKE_BUILD_PARALLEL_LEVEL=2
CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
.venv/bin/python -m pip wheel --no-build-isolation --no-deps ...
```

Questa è una configurazione, **non** un comando completo da eseguire: mancano lock degli artefatti, staging/installazione atomica e procedure di rollback. L'eventuale wheel costruito localmente sarà poi verificato per capacità di offload GPU e **inferenza CUDA reale** prima di dichiarare Ruth utilizzabile.

## 4. Criteri per proseguire

1. Nuovi test e preflight host confermati dall'utente.
2. Checkout acquisito **automaticamente** tramite `--prepare` con consenso iniziale, e commit verificati.
3. Lockfile con hash e sicurezza approvato.
4. Build CUDA in directory dedicata, mai installazione globale.
5. Validazione della libreria, GPU offload, caricamento di un modello e inferenza reale (fasi successive).

Il comando `--plan` non scarica e non installa; `--prepare` scarica i **soli sorgenti fissati** nella directory ignorata dal repository, ma non installa Python, non compila e non avvia CUDA. **Non usare `apt autoremove` per eliminare i residui NVIDIA 580** senza revisione separata.
