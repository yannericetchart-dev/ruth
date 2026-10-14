# Ruth — Fase 05: catalogo dinamico e selezione delle quantizzazioni

## Priorità architetturale corretta

La Fase 05 precede la decisione di scaricare o provare il modello di produzione.
Il default resta la **famiglia/versione Gemma 2 9B IT**; `Q4_K_M` è la
quantizzazione inizialmente registrata nel catalogo, **non** una scelta
automaticamente ottima né un profilo autorizzato su 6 GiB.

Lo scopo è scegliere il *candidato da verificare*, non pubblicare un
profilo con la sola dimensione del file GGUF.

## Fonte ed elenco disponibile

Fonte predefinita: Hugging Face. Per Gemma 2 9B IT:

- Originale: https://huggingface.co/google/gemma-2-9b-it
- Varianti GGUF: https://huggingface.co/bartowski/gemma-2-9b-it-GGUF

Il catalogo `catalog/model_catalog.json` continua a contenere **soltanto
varianti pinned autorizzate**. La discovery legge la risposta
`/api/models/{repo}/revision/main?blobs=true` per lo **stesso**
repository GGUF registrato, estrae tutti i file corrispondenti a
`{model_prefix}-{quant}.gguf` con **dimensione intera esatta e hash
LFS SHA-256**, blocca duplicati, revisioni/licenze diverse da un pin
preesistente e source gated. Non inventa dimensioni per file mancanti.
Non aggiorna il catalogo, non scarica modelli né accetta licenze.

```bash
./ruth models quantizations --purpose production --allow-metadata-network
```

La lista presenta **tutte le varianti risolte con LFS metadata**,
inclusi gli stati `architecture_or_unranked` (es. Q4_0 ARM) e
`cuda_support_verified=false`. Una quantizzazione trovata nel catalogo
remoto **non** è automaticamente supportata dall'eseguibile pinned:
la prova vera arriverà dopo pin e download separatamente autorizzati.

## Policy di qualità e memoria

Politica riproducibile e modificabile in
`catalog/quantization_policy.json`. L'ordine a priorità decrescente
segue descrizioni qualitative e classe di compressione pubblicate
dall'autore del GGUF, non un benchmark indipendente.
`quality_priority_heuristic` è **una regola ordinativa**, non
un punteggio di qualità misurata né una garanzia di prestazioni.
Q4_K_S, IQ4_XS e altri algoritmi non si possono confrontare con
precisione senza prove sullo stesso workload.

Il dispositivo viene inventariato **in sola lettura** con
`nvidia-smi --query-gpu=uuid,pci.bus_id,memory.total,memory.free`.
In presenza di GPU multiple/telemetria ambigua si blocca la
selezione. Il pre-filtro prende un budget provvisorio:

`budget_bytes = max(0, free_mib - reserve_mib) * 1048576`

Si propone una variante solo se la **dimensione GGUF intera in byte**
non supera questo budget. Il margine `reserve_mib` di default è
**768 MiB**, volutamente **modificabile dall'utente**; non equivale
alla misurazione della KV cache/compute buffer ed è un criterio
provvisorio, NON un budget GPU qualificato. L'uso dei byte su disco
è un proxy approssimato della dimensione dei pesi in GPU:
il runtime può allocare più memoria o scegliere layout diversi.

```bash
./ruth models recommend --purpose production --allow-metadata-network
./ruth models recommend --purpose production --allow-metadata-network --reserve-mib 1024 --top-k 5
```

La shortlist è ordinata per priorità di qualità euristica fra le sole
varianti che superano il **prefiltro per dimensione**. Riporta
`provisional_shortlist_ready`, `size_only_prefilter`,
`estimated_remaining_mib_after_file`, SHA256, revisione Git,
`full_gpu_residency_verified=false`,
`candidate_is_production_qualified=false`,
`download_started=false` e `catalog_modified=false`.

Se nessuna candidata soddisfa il budget, si restituisce
`no_quantization_passed_provisional_budget`. Non si modifica
automaticamente famiglia/modello, non si scarica una variante peggiore
di nascosto e **non** si abilita inferenza CPU per compensare VRAM
insufficiente.

## Dati reali forniti dall'utente — 14 ottobre 2026

- NVIDIA GeForce RTX 4050 Laptop GPU: **6141 MiB totali**,
  **5759 MiB liberi** nel momento dello snapshot (15 MiB `used`
  riportati; le grandezze non sommano al totale e sono da trattare
  come telemetria istantanea del driver, non come un bilancio contabile).
- Gemma 2 9B IT Q4_K_M: **5761057728 byte**,
  circa **5494,17 MiB** su disco e appena circa **265 MiB**
  tra byte del file e VRAM libera prima di qualunque altra allocazione.
- Revisione `d731033f3dc4018261fd39896e50984d398b4ac5`,
  SHA256 `13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe`,
  licenza `gemma`, `gated=false`. Questi valori sono stati
  confermati con la precedente chiamata `resolve` sul PC.

L'autore Hugging Face dichiara anche varianti **Q4_K_S (~5,48 GB)**,
**IQ4_XS (~5,18 GB)** e **Q3_K_L (~5,13 GB)**; le dimensioni
**esatte e gli hash** verranno ottenuti dal nuovo comando; non
copiamo una stima arrotondata dentro un pin.

La qualità percepita non è semplicemente il numero dei bit.
Il ranking è una prima politica dichiarata e dovrà essere sostituito/
calibrato con verifiche di qualità (e, ove disponibili, perplexity/
benchmarks rappresentativi) solo dopo aver trovato candidati che
possano effettivamente essere caricati sul dispositivo.

## Gate dalla Fase 05 alla Fase 06/07

La shortlist **non scarica** nulla. Per un candidato proposto:
1. Revision, filename, SHA256 LFS e byte devono essere **pinnati**
   in `catalog/model_catalog.json`, con modifica tracciata.
2. Termini della licenza Gemma vanno valutati e accettati
   esplicitamente, **separatamente** dall'autorizzazione al download.
3. Scaricare soltanto il candidato effettivamente approvato e
   verificare SHA256 locale.
4. Worker CUDA isolato con modello reale e KV cache/compute buffer,
   layer offloaded e memoria PID/GPU, n_ctx crescente e margine
   operativo verificato. Una GPU con memoria insufficiente deve
   portare a fallimento esplicito, **mai** a fallback CPU silenzioso.
5. Solo dopo le prove di qualità/prestazioni/contenuto si può
   qualificare `Gemma-2-9B-it` come profilo production.

Nessun nuovo wheel/installazione/compilazione si rende necessario.
`Kaggle` e la CVE `diskcache` rimangono TO DO.
