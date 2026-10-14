# Ruth — Catalogo modelli: Hugging Face e gerarchia selezione

## Decisioni iniziali

- **Provider predefinito:** Hugging Face (`huggingface`); metadata resolver implementato solo in lettura. Nessun download, login, libreria Python aggiuntiva o accettazione automatica della licenza.
- **Provider Kaggle:** `TO DO`, visibile nel registro ma non selezionabile finché autenticazione, metadati, permessi, formato e controlli SHA256 non sono implementati.
- **Famiglie iniziali:** `gemma`, `llama`, `qwen`, `mistral`. Qwen e Mistral sono **registrate senza candidati pre-validati**, non considerate mancanti dall'architettura.
- **Gerarchia dei modelli:** `family/model/version/format/quantization`. La quantizzazione è separata dal formato perché differenti GGUF dello stesso modello hanno requisiti di memoria distinti.

### Profili scelti

| Scopo | Modello | Candidato GGUF | Uso consentito |
|---|---|---|---|
| production | `gemma/gemma-2/gemma-2-9B-it/gguf/Q4_K_M` | `bartowski/gemma-2-9b-it-GGUF` | **Scelta di prodotto**, non validazione hardware |
| test | `llama/llama-3.2/llama-3.2-1B-instruct/gguf/Q4_K_M` | `bartowski/Llama-3.2-1B-Instruct-GGUF` | Test di funzionamento / integrazione **non sostitutivi** del profiling VRAM |

Il GGUF `gemma-2-9b-it-Q4_K_M.gguf` risulta intorno a **5,76 GB** e la versione Llama `Llama-3.2-1B-Instruct-Q4_K_M.gguf` intorno a **808 MB**, da confermare nella risposta API prima di autorizzare qualsiasi download. Queste dimensioni sono indicative, non garanzia di compatibilità effettiva. Su una RTX 4050 Laptop con **6 GiB** la quantizzazione Gemma `Q4_K_M` lascia poco margine per KV cache, compute buffer, framework e memoria già occupata; potrebbe richiedere un placement parziale sulla CPU oppure non funzionare entro i limiti fissati. **Nessun fallback CPU automatico**: se il test richiede offload completo e le risorse sono insufficienti, Ruth deve dichiararlo e proporre separatamente un'altra quantizzazione/contesto con consenso.

### Proprietà e licenze

I GGUF indicati sono **quantizzazioni ospitate da un soggetto terzo**; non vanno confusi con i repository dei produttori originali:

- Google Gemma 2 9B IT originale: https://huggingface.co/google/gemma-2-9b-it — licenza `gemma`, accettazione delle condizioni richiesta sull'originale.
- Quantizzazione Gemma: https://huggingface.co/bartowski/gemma-2-9b-it-GGUF
- Meta Llama 3.2 1B Instruct originale: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct — licenza `llama3.2`, accesso al repository originale soggetto a condizioni.
- Quantizzazione Llama: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

La presenza di un file GGUF pubblico **non** elimina condizioni d'uso o attribuzioni dei modelli originali. Ruth registra licenza e stato `license_acknowledgement_required=true`; nessun flag di selezione è un consenso legale.

### Comandi disponibili

Tutte le operazioni di catalogo sono indipendenti dalla GPU, non importano `llama_cpp` e non modificano la `.venv`.

```bash
./ruth models list
./ruth models select --purpose test
./ruth models select --purpose production
./ruth models select --purpose test --family gemma --model gemma-2 --version gemma-2-9B-it --format gguf --quantization Q4_K_M
```

`select` è **completamente offline**. Riporta repo/model, formato, quantizzazione, SHA-256 già registrato e stato da verificare, ma `downloadable=false` e `local_model_ready=false`.

Per risolvere revisione remota (hash Git immutabile, metadati LFS SHA256 e dimensione) occorre un'autorizzazione **solo ai metadata**:

```bash
./ruth models resolve --purpose test --allow-metadata-network
```

La richiesta consulta esclusivamente `https://huggingface.co/api/models/{repo}/revision/main?blobs=true`, rifiuta redirect esterni e confronta il digest remoto col pin locale del catalogo. Se i metadati sono coerenti, restituisce `source_metadata_verified_license_pending`, `revision` SHA Git 40 caratteri, dimensione e `sha256`. **Non** trasferisce GGUF, non salva token e non accetta licenze. Il commit di `main` è bloccato per il successivo download: mai usare un ramo mobile senza prima risolverlo.

Il catalogo è in `catalog/model_catalog.json`, validato dal modulo standard-library `ruth_runtime/model_catalog.py`. Il resolver fallisce in presenza di family/variant non registrate, hash o licenza incompatibili, file assente, metadata LFS incompleti, revisione non immutabile o variazione del pin. Nessuna sostituzione silenziosa di modello, quantizzazione o backend.

## Cosa non è ancora stato implementato

- Download dei file GGUF, autenticazione/tokens per modelli gated, licenza approvata, caching/checkpoint e recupero da errori: **Fase 06 da implementare con autorizzazioni separate**.
- Analisi automatica di tutte le varianti GGUF e compatibilità reale: Fasi 04–07.
- Inference worker, token generati, misurazioni di offload GPU/VRAM: **non verificati**.
- Ulteriori modelli Qwen, Mistral e altri repository: catalogo espandibile, ma nessuna selezione fittizia.
- Kaggle: provider registrato come **TO DO**.

Il provider Hugging Face definisce il flusso predefinito, non equivale a
un download implicito. La vulnerabilità `diskcache` resta **TO DO non prioritario**,
come deciso.
