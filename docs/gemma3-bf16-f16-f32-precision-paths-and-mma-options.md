# Ruth / Gemma 3 — Percorsi di precisione BF16, F16, F32 e opzioni per FlashAttention MMA BF16

**Stato:** documento di analisi e piano di ricerca; nessuna modifica al runtime di produzione.  
**Data:** 2026-10-15  
**Ambito verificato:** Gemma 3 1B IT, GGUF BF16 con SHA256 `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`, backend llama.cpp fissato al commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`, GPU NVIDIA Ada 6 GiB, esperimenti Phase 28.

## 1. Obiettivo

Comprendere **quali precisioni vengono effettivamente usate** da Ruth nelle diverse fasi e operazioni di inferenza: non dedurle dal nome del modello, da un parametro KV o dal nome di un kernel. Analizzare l'eventuale sostituzione del percorso di prefill che converte BF16 in F16 con un FlashAttention MMA BF16 nativo; confrontare rigorosamente prestazioni, memoria e qualità numerica prima di qualsiasi modifica.

Le domande principali sono:

1. Quali pesi del GGUF sono davvero BF16, F16, F32 o altri formati? Quali sono i dtypes delle attivazioni e dei risultati intermedi?
2. Dove sono eseguite conversioni (BF16→F16, F16→F32, BF16→F32, altre) e dove sono allocati i buffer temporanei?
3. Quale precisione entra nell'operazione aritmetica e **in quale precisione vengono accumulati i risultati** (ad es. input BF16 e accumulazione F32)?
4. Perché il dispatcher usa MMA_F16 nel prefill e VEC_BF16 nel decode? Quali altre configurazioni selezionano questi o altri kernel?
5. È possibile integrare o sperimentare un FlashAttention **MMA con operandi BF16** evitando conversioni e preservando la semantica numerica?
6. Quali differenze di latenza, VRAM, divergenza dei logits, stabilità e copertura funzionale ne deriverebbero?

## 2. Distinzioni terminologiche: non confondere quattro precisioni

| Dimensione | Definizione | Esempio | Stato attuale |
|---|---|---|---|
| Precisione dei pesi del modello | Formato di memorizzazione dei tensori GGUF | `blk.0.attn_q.weight` | **Inventario per tensore ancora da eseguire**; il catalogo seleziona GGUF `BF16` ma non prova il dtype di ogni tensore |
| Precisione della cache KV | Tipo degli elementi K/V persistenti nel contesto | K=BF16, V=BF16 | **Verificato** nel candidato `bf16-on`, 52 MiB logici di KV a `n_ctx=2048` |
| Precisione di operandi/intermedi | Tipi letti dal kernel e dei suoi buffer temporanei | cache BF16 → K/V temporanei F16 durante MMA | **Verificata** per il percorso Flash prefill; da censire nelle altre operazioni |
| Precisione di accumulazione/output | Tipo con cui si accumulano somme o si producono output | F32 nel risultato di FlashAttention; accumulazione F32 in alcune istruzioni MMA | **Parzialmente verificata nel sorgente**, da attribuire per operazione/dispatch effettivo |

### 2.1 I 'layer BF16' non sono un contratto aritmetico BF16

Il file GGUF del modello è identificato nel catalogo come BF16; i pesi principali delle proiezioni sono attesi prevalentemente BF16, ma **non si deve affermare che ogni singolo tensore o operazione dell'attenzione sia BF16** senza enumerare effettivamente i tipi GGUF. Norm, bias, embedding/output e tensori ausiliari possono avere scelte proprie. Il tipo dei pesi non determina automaticamente il tipo delle attivazioni, degli accumulatori o del risultato del kernel.

**Azioni:** inventariare il GGUF locale in sola lettura con il parser già ammesso/controllato nel progetto; produrre conteggi per dtype e un elenco puntuale di tutti i tensori `attn_*`, `ffn_*`, `norm_*`, embedding e output, includendo dimensioni e numero di elementi. Conservare SHA256 e percorso di origine. Non leggere o scaricare modelli esterni.

## 3. Evidenze già raccolte (senza nuove esecuzioni GPU)

### 3.1 Candidato BF16/BF16 Flash ON a `n_ctx=2048`

- Native offload osservato: **27/27 layer**; KV K=BF16 e V=BF16; 52 MiB logici a contesto 2048.
- Il prefill usa **`FLASH_MMA_F16` + `FLASH_FIXUP`** e il decode usa **`FLASH_VEC` (K/V BF16) + `FLASH_COMBINE`**. Questa non è un'ipotesi: i kernel sono stati identificati via Nsight e attribuiti tramite NVTX/CUPTI.
- La traccia a granularità di nodo rivela 128 CUDA Graph replay e 32 intervalli con attenzione lanciata direttamente. **160/160** intervalli host prefill/decode classificati senza ambiguità.
- Nell'acquisizione node-level sono stati correlati **81.008 record kernel interni ai grafi** alla chiamata `cudaGraphLaunch` e al range NVTX corrispondente: graphId 2 → prefill (796 identificatori di nodo-kernel × 68 replay), graphId 10 → decode (448 × 60). I set di nodi-kernel osservati sono completi per ogni replay registrato.
- Contabilità dei kernel Flash principali **osservati**: prefill 728 diretti + 1.768 nei graph = **2.496 = 96 × 26**; decode 104 diretti + 1.560 nei graph = **1.664 = 64 × 26**. I rispettivi helper FIXUP/COMBINE seguono lo stesso conto.
- Le operazioni di conversione BF16→F16 sono supportate sia dal sorgente sia dall'evidenza dei kernel `convert_unary_cont_vec4<__nv_bfloat16,__half>`: nelle precedenti tracce graph-level sono stati riconosciuti **1.456 lanci diretti**, ossia 2 × 728. **Resta da censire esplicitamente la famiglia e il costo delle conversioni dentro i CUDA Graph**, senza ricavarli automaticamente dal nome o dal numero dei main kernel.
- Il benchmark BF16/ON ctx2048 con NVTX riporta, indicativamente, prefill mediano **331,2395 ms**, decode totale mediano **207,076 ms** per 16 token, **77,27 tok/s**. Le tracce `--cuda-graph-trace=node` hanno overhead e **non sono benchmark temporali comparabili**.
- I logits dell'esecuzione campionata BF16/ON hanno SHA256 `5629cf4d0e7437e12af1ec9412309e60e76192d85c5e7a822dcd4c7695e613d0`, coerente con la baseline precedente. L'artefatto contiene **solo due vettori di logits**; non dimostra equivalenza numerica su tutti i token o prompt.

Il rapporto fra BF16/ON e F16/OFF è **confondente**: cambia contemporaneamente dtype della KV e politica Flash. Il piccolo vantaggio BF16/ON osservato su questo workload non prova il contributo isolato di BF16 né quello di FlashAttention.

### 3.2 Perché il prefill converte BF16 → F16?

Nel **commit fissato**, `ggml/src/ggml-cuda/fattn.cu` definisce i kernel selezionabili `NONE`, `TILE`, `VEC` e `MMA_F16`, ma non `BEST_FATTN_KERNEL_MMA_BF16`. Quando viene selezionato `BEST_FATTN_KERNEL_MMA_F16`, il codice imposta `need_f16_K=true` e `need_f16_V=true`. In `fattn-common.cuh`, se K/V non sono già F16, la routine alloca/usa spazio temporaneo e lancia le conversioni per alimentare MMA_F16. **La cache KV originale rimane BF16**: la conversione non ne cambia il formato persistente.

Riferimenti esatti:

- [fattn.cu: enum e selezione kernel](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn.cu#L516-L665)
- [fattn.cu: need_f16_K/V](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn.cu#L725-L770)
- [fattn-common.cuh: buffer conversione](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn-common.cuh#L1010-L1090)

### 3.3 Perché il decode può mantenere K/V BF16?

Il dispatcher può selezionare `BEST_FATTN_KERNEL_VEC` per un token/query e per le condizioni della GPU Ada e del layout delle head; **non è una regola generale valida per ogni decode**. Il codice include istanze vettoriali specifiche `FATTN_VEC_CASES_ALL_D(BF16,BF16)`: K/V possono essere letti direttamente in BF16. Il kernel vettoriale usa anche registri e accumuli in altri tipi (es. `float` / `float2`); esistono conversioni locali all'interno del kernel. **Nessuna conversione globale preventiva BF16→F16 K/V** è richiesta per il percorso BF16/BF16 VEC osservato, ma ciò non significa calcoli esclusivamente BF16.

- [fattn.cu: selezione VEC e casi BF16/BF16](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn.cu#L419-L515)
- [fattn-vec.cuh: lettura BF16 e percorsi interni](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn-vec.cuh#L87-L150)
- [fattn-common.cuh: prodotto K×Q BF16 e V BF16](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn-common.cuh#L116-L143)

### 3.4 Esiste un FlashAttention MMA BF16?

**Come capacità hardware, sì:** Ada può usare Tensor Core con operandi BF16; il codice `ggml/src/ggml-cuda/mma.cuh` contiene istruzioni PTX `mma.sync...f32.bf16.bf16.f32`, cioè operandi BF16 e accumulazione F32. **Come kernel FlashAttention selezionabile nel backend CUDA del commit fissato, no:** qui il percorso disponibile è MMA_F16; l'istruzione generica BF16 non dimostra l'esistenza di un kernel Flash MMA_BF16 completo, supportato e ottimizzato. Implementazioni esterne possono supportare BF16, ma non sono integrate automaticamente in Ruth.

- [mma.cuh: MMA BF16 con accumulazione F32](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/mma.cuh#L1238-L1253)

## 4. Indagine richiesta: dove Ruth usa F32 o altre precisioni?

**Un inventario completo del percorso numerico è un deliverable obbligatorio** prima di selezionare varianti, non un controllo opzionale. L'esistenza di una dichiarazione `float` nel codice prova una rappresentazione nel sorgente, non automaticamente ogni istruzione e il relativo formato eseguito dalla GPU.

| Sottosistema da censire | Domande misurabili / tipi da distinguere | Evidenza o punto di partenza |
|---|---|---|
| Pesi e metadata GGUF | BF16/F16/F32 per Q/K/V/O, norm, MLP, embedding, output; mixing dei tensori | Inventario reale dell'indice GGUF con hash verificato; ancora da eseguire |
| Embedding, proiezioni Q/K/V/O, MLP e output | Tipo dei pesi, tipo degli input/attivazioni, precisione GEMM, accumulatori e output; eventuali cast impliciti | Traccia del grafo GGML + operatori CUDA/CUTLASS, non attribuire genericamente tutti i matmul all'attenzione |
| Normalizzazioni RMSNorm, residui/add, scale, RoPE | Dtype dei tensori di ingresso/uscita; conversioni, operazioni float e fp32 nei kernel | `src/llama-graph.cpp`, `ggml/src/ggml-cuda/` e trace delle singole famiglie |
| Prefill FlashAttention MMA_F16 | BF16 cache → buffer F16, possibile casting Q, precisione del prodotto, softmax online, accumulazione, output | `fattn.cu`, `fattn-common.cuh`, `fattn-mma-f16.cuh`, `mma.cuh`; Nsight |
| Decode FlashAttention VEC_BF16 | K/V BF16, tipi di Q e degli accumulatori, `float`/`float2`, conversioni nei registri e output | `fattn-vec.cuh`, `fattn-common.cuh`; Nsight |
| Score Q·K, maschera, softmax e normalizzazione | Dtype effettivo di exp/max/somma, scaling e softcap; controllare precisione locale e accumulo | Kernel CUDA e varianti per template eseguite |
| KV update, conversione, views e SWA | Dove K/V vengono scritti in BF16, copie temporanee F16, F32/F16/BF16 dei nodi SWA e global | log allocazione KV + grafo nativo + Nsight; non inferire dal solo dtype KV |
| Output finale e logits | Dtype delle attivazioni prima di `lm_head`, GEMM/output, logit buffer `float32`, eventuali cast/normalizzazioni | Distingui dtype dell'artefatto NumPy dalla precisione di calcolo; `llama_get_logits` e output del worker |
| CUDA Graph | Tipo di ogni nodo (kernel, memcpy, memset, eventi, altre operazioni); conversioni e formati in ciascun replay | Nsight `--cuda-graph-trace=node`; dati osservati non provano la copertura di tutti i tipi di nodo |

Un punto già confermato nel sorgente: `fattn-common.cuh` contiene `GGML_ASSERT(KQV->type == GGML_TYPE_F32)` nel percorso di lancio di FlashAttention e il kernel BF16 vettoriale usa accumulatori `float`. Questo **non implica** che ogni attivazione, ogni prodotto e tutta la softmax siano eseguiti in F32: serve attribuzione specifica al kernel compilato ed eseguito.

Il deliverable sarà una **matrice di provenienza delle precisioni per operazione** con: fase; layer (global/SWA); op GGML; dtype pesi; dtype input/Q/K/V; eventuale conversione e buffer; kernel GPU; dtype operand; dtype accumulator; dtype output; fonte (GGUF/sorgente/trace/PTX); evidenza verificata o inferita; incertezza residua.

## 5. Piano di lavoro, ordinato per rischio crescente

### A. Inventario statico e offline — priorità immediata

- Verificare nuovamente SHA256 del GGUF e wheel nativo, senza download né inferenza.
- Esportare una tabella reale dei tensori GGUF, dei loro dtype, shape e dimensioni; verificare in particolare Q/K/V/O, MLP, norm, embedding e output.
- Creare una mappa dei cast e dtype dal modello GGML e dal codice CUDA del commit fissato; **non** attribuire automaticamente `GGML_TYPE_F32` di un output al resto del calcolo.
- Dalle due tracce Nsight già acquisite, classificare le conversioni BF16↔F16/F32, i kernel Flash e le altre famiglie, separando host direct e graph replay tramite correlationId/graphNodeId.
- Contare kernel, somme dei tempi GPU (non tempi di prefill/decode), e buffer temporanei. Evitare doppio conteggio e ricostruzioni fondate sulla sola uguaglianza numerica.
- Per i 26 layer (4 global, 22 SWA) distinguere percorsi e tipologie di attenzione; segnalare le porzioni non osservabili.

### B. Prove numeriche offline e copertura

- Analizzare gli artefatti logits già esistenti imponendo stesso modello, contesto, tokenizzazione e sequenza teacher-forced; calcolare max/mean abs error, relative error con soglie sensate, KL/JSD, top-1/top-k e differenze di ranking.
- Dichiarare esplicitamente che due vettori di logits per prova **non** coprono il comportamento reale su prompt/tokens differenti; progettare poi un workload numerico più rappresentativo solo con nuova autorizzazione esplicita.
- Cercare overflow/underflow, NaN/Inf e differenze BF16→F16 per valori estremi; distinguere un possibile rischio teorico dall'incidenza reale nel workload.
- Non usare una precisione 'maggiore' come ground truth per definizione: eventuale F32 deve essere confrontato con limiti noti di algoritmo, accumulazione e stabilità.

### C. Completare la matrice Phase 28 già approvata

Le configurazioni attualmente previste sono `f16-off` (KV F16/F16, Flash OFF), `bf16-on` (KV BF16/BF16, Flash ON), `f32-off` (KV F32/F32, Flash OFF), `mixed-on` (KV BF16/F16, Flash ON), per i contesti 256, 512, 1024 e 2048. **Non estendere automaticamente il gate, i candidati, i contesti o gli offload.** Valutare i profili mancanti con gli stessi controlli di integrità modello, VRAM, offload, logits e ripetibilità.

Importante: questa matrice **non isola** il contributo della precisione e di FlashAttention. Per un confronto fattoriale servirebbero eventuali candidati strettamente controllati `f16-on` e `bf16-off` (o equivalenti), introdotti come **nuovo esperimento esplicitamente autorizzato**, non come modifica nascosta ai gate esistenti.

### D. Prototipi MMA BF16 nativo — solo se i dati ne giustificano il costo

Opzioni da esaminare senza assumere in anticipo superiorità:

1. **Baseline:** mantenere KV BF16 + MMA_F16 con conversione temporanea. Quantificare con precisione costo conversioni (inclusi replay) e costo MMA.
2. **KV F16 con Flash ON:** eliminerebbe alcune conversioni ma modificherebbe range numerico, persistenza e potenziali output; nuovo candidato necessario e non automaticamente migliore.
3. **MMA BF16 nativo:** implementare o adattare un kernel Flash che legga K/V BF16 e usi Tensor Core BF16, con controllo esplicito della precisione d'accumulo, maschera, softcap, GQA, SWA/global, forme head e CUDA Graph; non basta sostituire `half` con `nv_bfloat16`.
4. **Backend Flash alternativo:** valutare compatibilità, licenze, build riproducibile, dimensioni delle head, layout GGUF/GGML, driver, VRAM e profiling; evitare dipendenze non approvate.
5. **Ibrido:** scegliere BF16 nativo solo nei casi in cui la forma del problema e le misure dimostrino un vantaggio, mantenendo fallback verificati. Necessario controllo di regressione e chiarezza di attribuzione per fase.

Il criterio è **misurare** il trade-off. Input BF16 + accumulazione F32 non garantisce automaticamente logits più accurati o tempi migliori rispetto a input convertiti F16; la conversione BF16→F16 non restituisce bit di mantissa già persi e può introdurre problemi di intervallo dinamico.

### E. Metriche e gate prima di proporre una produzione

- Benchmark differenziati prefill/decode con mediana e dispersione di run ripetuti, workload e output-equivalenza invariati, con e senza tracciamento; i tempi raccolti con tracing node-level restano diagnostici.
- Memoria reale con peak VRAM osservata; distinzione fra peso modello, KV persistente, spazio temporaneo conversione, scratch e picco driver.
- Correlazione NVTX/CUPTI dei kernel reali (direct + replay), conversioni, andamenti per lunghezza contesto, output per token e varianti di head.
- Robustezza numerica documentata, QA di generazione multi-token, regressioni di qualità, eccezioni e `NaN/Inf`; limiti e copertura del campionamento espliciti.
- Nessun benchmark 'vincitore' se varia simultaneamente tipo KV e Flash senza prove di separazione degli effetti.
- Qualsiasi cambiamento di runtime produttivo richiede nuova autorizzazione e gate indipendenti; **`production_qualified=false`** fino a quel momento.

## 6. Deliverable e definizione di conclusione

1. `precision_inventory` offline: dtype di ogni tensore rilevante del GGUF e statistiche per tipo.
2. `precision_execution_map`: per-op dispatch e catena dei cast inclusi FP32, con evidenze e livello di certezza.
3. `conversion_cost_report`: tempi GPU aggregati attribuiti alle conversioni direct/graph, numero chiamate ed effetto sull'uso di buffer temporanei, senza confondere tempi kernel con tempo end-to-end.
4. `precision_numeric_report`: confronti numerici con campioni, metriche, hash, limiti e copertura documentati.
5. `mma_bf16_feasibility`: schema tecnico per un kernel BF16 nativo, test necessari, costi, benefici plausibili e rischi; decisione **go/no-go** motivata, non una promessa di ottimizzazione.

**Primo passo operativo:** inventario GGUF + mappatura statica dei tipi nel codice e nei trace SQLite locali, in sola lettura; solo successivamente decidere quali inferenze aggiuntive siano necessarie. Nessuna modifica automatica a produzione o policy.

## 7. Prima implementazione offline — GGUF dtype e conversioni Nsight

Sono disponibili due strumenti nuovi e isolati (nessuna importazione di
`llama_cpp`, nessuna allocazione CUDA e nessuna modifica alla policy
di produzione):

1. `ruth_runtime.gemma3_phase28_precision_inventory` analizza
   **solo l'intestazione, i metadati e i descrittori dei tensori GGUF v3**
   senza caricare i pesi. Il modello è selezionato esattamente dal catalogo
   production **solo per identificare l'artefatto già esistente**;
   dimensione e SHA256 vengono confrontati ai valori pinned prima che
   l'inventario sia dichiarato verificato. Calcola conteggi per dtype
   (BF16, F16, F32, altri) e categorie (attention_projection,
   attention_norm, feed_forward, normalization, embedding, output, ...).
   La versione del layout dei tipi GGML è fissata al commit llama.cpp
   `0c1e5709`. Anche i dtype sconosciuti sono indicati esplicitamente
   come `UNKNOWN_N` e **non** reinterpretati come BF16.
2. `ruth_runtime.gemma3_phase28_precision_trace` legge gli export SQLite
   già acquisiti, riconosce i kernel `convert_unary*` con tipi sorgente
   e destinazione espliciti nel simbolo CUDA, ne conta i record e somma
   le durate GPU. Quando la traccia node-level contiene gli ID corretti,
   usa il preesistente **audit causale** `gemma3_phase28_node_audit`
   per attribuire le conversioni dei CUDA Graph alla fase NVTX.
   Per i lanci diretti usa, dove possibile, `correlationId` CUDA
   Runtime + intervallo NVTX della stessa CPU thread. Riporta i kernel
   non classificati più onerosi e gli operatori con nome F32
   riconosciuto, **senza** affermare che il nome descriva il dtype di
   ogni registro/operazione.

Eseguire dalla root del repository, dopo `git pull --ff-only`:

```bash
.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_precision_inventory \
  tests.test_phase28_gemma3_precision_trace -v

# Legge solo GGUF locale. L'SHA256 richiede la lettura sequenziale
# dell'intero file, ma non carica il modello né alloca VRAM.
.venv/bin/python -m ruth_runtime.gemma3_phase28_precision_inventory

# Include nomi, dimensioni, dtype e offset di TUTTI i tensori:
.venv/bin/python -m ruth_runtime.gemma3_phase28_precision_inventory \
  --all-tensors

# Analizza la traccia CUDA Graph a granularità di nodo senza esecuzione GPU:
.venv/bin/python -m ruth_runtime.gemma3_phase28_precision_trace \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048

# Stessa indagine nella traccia a livello di grafo:
.venv/bin/python -m ruth_runtime.gemma3_phase28_precision_trace \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite \
  --candidate bf16-on --n-ctx 2048
```

I comandi non scaricano nulla e sono in **sola lettura**: il modello
resta quello già presente e il database SQLite è aperto con `mode=ro`.
Gli script non creano file di output se non richiesti tramite
redirezione shell. I test usano GGUF e SQLite sintetici temporanei.

Campi diagnostici più importanti:

- GGUF: `status=pinned_gguf_precision_inventory_verified`,
  `tensor_dtype_counts`, `tensor_categories`,
  `non_bf16_tensor_sample`, e `tensors` con `--all-tensors`.
- Nsight: `kernel_name_survey_verified`,
  `conversion_kernel_records`, `conversion_summary[*].source_dtype`,
  `target_dtype`, `kernel_gpu_ms_sum`, `by_phase`, `by_graph_id`,
  `other_precision_name_evidence`, `top_other_kernel_names`.
- Le conversioni riconosciute con dtype non estraibile restano
  `conversion_kernel_template_unparsed` invece di essere ignorate.
- La correlazione diretta o grafica mancante è esplicita in
  `unattributed_direct_conversion_records` e
  `unattributed_graph_conversion_records`.
- **Sempre falso:** `source_precision_runtime_inventory_complete`
  e `production_qualified`. Nessuna informazione sui soli nomi
  dei kernel certifica la precisione interna di tutte le operazioni.

**Stato:** implementazione pubblicata sul branch sperimentale,
non ancora eseguita contro il GGUF e gli SQLite locali dell'utente.
Le metriche reali sono in sospeso fino all'esecuzione sul suo host.

## 8. Riferimenti interni

- [Fase 27 — precisione e benchmark](phase-27-gemma3-precision-and-kernel-benchmark.md).
- [Fase 28 — prove CUDA/NVTX e contesti](phase-28-gemma3-dispatch-and-context-scaling.md).
- [Phase 28 gate](../ruth_runtime/gemma3_phase28_gate.py) e [worker](../ruth_runtime/gemma3_phase28_worker.py).
- [Audit dei nodi CUDA Graph](../ruth_runtime/gemma3_phase28_node_audit.py).
- Codice sorgente pinned `ggml-org/llama.cpp@0c1e57098bba43ac29e6e3b677cdceebdd22334f` e sorgenti CUDA collegati nelle sezioni precedenti.
