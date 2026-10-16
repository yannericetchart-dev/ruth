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

## 8. Evidenza reale dal primo inventario GGUF e dal trace conversioni

Il 15 ottobre 2026 l'utente ha eseguito gli strumenti offline sul GGUF
pinned e su entrambe le tracce Nsight a contesto 2048. Il test riportato
ha concluso `Ran 5 tests ... OK`.

**GGUF verificato per SHA256**:
`b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`,
versione 3, **340 tensori** e file da **2.006.573.568 byte**.

| Tipo e categoria | Tensori BF16 | Tensori F32 | Elementi BF16 | Elementi F32 |
|---|---:|---:|---:|---:|
| Proiezioni Q/K/V/O | 104 | 0 | 76.677.120 | 0 |
| Normalizzazione attenzione | 0 | 78 | 0 | 43.264 |
| Embedding | 1 | 0 | 301.989.888 | 0 |
| Feed forward | 78 | 26 | 621.084.672 | 29.952 |
| Altre normalizzazioni | 0 | 53 | 0 | 61.056 |
| **Totale** | **183** | **157** | **999.751.680** | **134.272** |

In base ai **parametri**, la quota F32 è **0,0134287%**,
non il 46,2% dei tensori. In byte: BF16 1.999.503.360,
F32 537.088. Con l'offset del payload GGUF di 6.533.120 byte,
la somma è **esattamente** la dimensione del file
(2.006.573.568 byte). **La contiguità di tutti i tensori è stata ora verificata** sul
GGUF locale: `tensor_storage_accounting_complete=true`,
`tensor_storage_layout_contiguous_verified=true`,
`tensor_storage_bytes_accounted=2000040448`, e
`gguf_tensor_data_region_bytes=2000040448`.
Questi controlli non analizzano i valori numerici dei pesi.

**Nsight `--cuda-graph-trace=node`**:
105.128 record kernel totali, di cui 81.008 negli CUDA Graph
e 24.120 registrati direttamente. Tutte le conversioni di seguito
sono attribuite al **prefill** nella stessa traccia; la relazione
causale per i kernel nei graph era già stata verificata.

| Cast (kernel `convert_unary_cont_vec4`) | Tutti i record | Dentro i graph | Direct | Somma tempi GPU |
|---|---:|---:|---:|---:|
| `float → __nv_bfloat16` (F32→BF16) | 17.184 | 12.172 | 5.012 | 28,6635 ms |
| `__nv_bfloat16 → __half` (BF16→F16) | 4.992 | 3.536 | 1.456 | 24,7789 ms |
| **Totale** | **22.176** | **15.708** | **6.468** | **53,4424 ms** |

I conteggi sono divisibili precisamente per i 96 intervalli di
prefill: **179 cast F32→BF16 e 52 cast BF16→F16 per intervallo
in media**. Inoltre i conteggi *all'interno dei grafi* corrispondono
esattamente a **179 e 52 kernel per replay** dei 68 replay
di prefill; i conteggi *direct* a **179 e 52 per ciascuno dei
28 intervalli in media**. È consistente con due conversioni
BF16→F16 (K e V) per ciascuno dei 26 livelli di attenzione.
L'identificazione causale di tutti i 179 cast F32→BF16 ai
nodi GGML originari **non è ancora verificata**.

La vecchia traccia graph-level esponeva soltanto 6.468 conversioni
dirette (5.012 F32→BF16 + 1.456 BF16→F16) e quindi
sottocontava intenzionalmente le conversioni riprodotte nei graph.
Le durate GPU sono somme di kernel per l'intero workload
(1 warmup + 3 ripetizioni); non sostituiscono latenza end-to-end,
non isolano automaticamente il costo di un ipotetico MMA BF16
e **non vanno confrontate come tempi equivalenti fra acquisizioni
graph-level e node-level**.

### Evidenza indipendente FP32 al di fuori dei soli cast

Nel medesimo trace a livello di nodo sono comparsi kernel
`rms_norm_f32`, `rms_norm_mul_rope_f32`,
`unary_gated_op_kernel<&op_gelu,float>`,
`scale_f32<float>`, `mul_mat_vec_f<__nv_bfloat16,float,...>`,
`k_set_rows<float,long,__nv_bfloat16>` e
`ampere_s16816gemm_bf16_*`. I nomi danno evidenza
di tipizzazione e famiglie aritmetiche diverse; **non** provano
per ciascuna di esse il dtype degli accumulatori o tutta la
sequenza numerica. La mappatura operazione GGML→kernel resta
l'indagine successiva.

### Nuova diagnostica per classificare queste famiglie

`gemma3_phase28_precision_trace` ora fornisce:
`arithmetic_kernel_family_summary`, `conversion_by_path`,
`direct_conversion_phase_correlation_verified` e
`arithmetic_precision_coverage_complete=false`. Ogni famiglia
riporta `accumulator_dtype_verified=false`: il dtype non può
essere dedotto da un semplice nome CUDA.

Anche il GGUF inventory ora include
`tensor_dtype_element_counts`, `tensor_dtype_storage_bytes`,
`tensor_storage_layout_contiguous_verified` e i byte contabilizzati.

Comandi di verifica sullo stesso host, **senza CUDA**:

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_precision_inventory \
  tests.test_phase28_gemma3_precision_trace -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_precision_inventory

.venv/bin/python -m ruth_runtime.gemma3_phase28_precision_trace \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048
```

**Conclusione sperimentale provvisoria:** modello quasi interamente
BF16 per parametri, con pesi di normalizzazione F32; prefill
comprende conversioni esplicite F32→BF16 e BF16→F16; decode VEC
legge K/V BF16. Occorre ancora attribuire esattamente gli operatori
GGML e verificare la precisione effettiva degli accumulatori.
Qualificazione produzione sempre falsa.

## 9. Secondo output reale: famiglie aritmetiche FP32/BF16 e source audit

Nuovo output ottenuto sul medesimo modello e sulla traccia
`bf16-on-ctx2048-nvtx3-nodes.sqlite` (nessuna inferenza nel lettore).
L'inventario verificato riporta ora espressamente:
`tensor_dtype_element_counts={"BF16":999751680,"F32":134272}`,
`tensor_dtype_storage_bytes={"BF16":1999503360,"F32":537088}`,
`tensor_storage_layout_contiguous_verified=true`, e
`tensor_storage_accounting_complete=true`.
Le seguenti durate rappresentano **somme di kernel GPU registrati
nella traccia node-level**, non latenze complete né numeri da confrontare
con i tempi della traccia graph-level.

| Famiglia nome CUDA (dtype indiziati, non accumulazione dedotta dal nome) | Kernel registrati | Tempo GPU sommato | Prefill | Decode |
|---|---:|---:|---:|---:|
| `GEMM_BF16_NAMED` | 17.184 | 888,4698 ms | 17.184 | 0 |
| `MATVEC_BF16_F32_NAMED` | 10.060 | 749,2876 ms | 12 | 10.048 |
| `RMS_NORM_F32` | 16.432 | 109,4021 ms | 9.712 | 6.720 |
| `GELU_F32_NAMED` | 2.400 | 29,3442 ms | 2.400 | 0 |
| `RMS_NORM_ROPE_F32` | 8.320 | 26,6055 ms | 4.992 | 3.328 |
| `CUBLASLT_SPLITK_REDUCE` | 7.392 | 18,3250 ms | 7.392 | 0 |
| `SET_ROWS_F32_BF16_NAMED` | 8.320 | 12,9774 ms | 4.992 | 3.328 |
| `SCALE_F32` | 4.320 | 9,2341 ms | 2.592 | 1.728 |
| `GATHER_F32_NAMED` | 136 | 0,2359 ms | 8 | 128 |
| `ADD_F32_NAMED` | 68 | 0,1074 ms | 4 | 64 |

L'audit riporta ora **sia** `direct_conversion_phase_correlation_verified=true`
**sia** `graph_phase_causal_mapping_verified=true`. I 22.176
lanci di conversione si suddividono in:

- `direct`: 6.468 kernel, 16,3676 ms sommati;
- `graph_node`: 15.708 kernel, 37,0748 ms sommati.

Le conversioni restano per intero nel prefill. Questo **non**
certifica tutte le operazioni interne in FP32:
`arithmetic_precision_coverage_complete=false` e
`accumulator_dtype_verified=false` per ciascuna famiglia nel report.

### 9.1 Spiegazione sorgente verificata per alcuni percorsi

La revisione **immutabile** di llama.cpp
`0c1e57098bba43ac29e6e3b677cdceebdd22334f` contiene
evidenze più forti di un semplice suffisso nel nome CUDA:

- **Matmul/cuBLAS BF16, compute F32**: in
  [ggml-cuda.cu, specializzazione `batched_mul_mat_traits<BF16>`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1376-L1392)
  il tipo degli operandi è `CUDA_R_16BF`, mentre
  `compute_type=CUBLAS_COMPUTE_32F`. In
  [ggml-cuda.cu, preparazione operandi](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1432-L1491)
  gli input non-BF16 vengono convertiti nel tipo scelto.
  La gestione dell'output su NVIDIA Ada preferisce F32
  [ggml-cuda.cu, output `prefer_f32_output`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1491-L1528).
  **Il numero di kernel `GEMM_BF16_NAMED` coincide esattamente
  con quello dei cast F32→BF16 (17.184)**, e nel graph 2
  coincidono anche i 12.172 record. Questo è *fortemente
  coerente* con una conversione F32→BF16 dell'operando del GEMM
  per lancio, ma non prova ancora l'identità uno-a-uno dei
  relativi nodi GGML né l'intero percorso di ciascun matmul.
- **RMSNorm FP32**: la funzione
  [`rms_norm_f32` in norm.cu](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/norm.cu#L77-L157)
  riceve `float*`, usa `float tmp`, riduzione e
  `rsqrtf`: per questo kernel la **riduzione FP32 è
  supportata dal sorgente**, non solo dal nome.
- **RMSNorm + RoPE FP32, possibile output di altro tipo**:
  [`rms_norm_mul_rope_f32` in rope.cu](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/rope.cu#L710-L788)
  legge `const float *x`, calcola `float tmp` e
  usa `ggml_cuda_cast<D>` per l'output. **Il dtype D
  deve essere letto dall'istanza effettiva**, non
  dedotto dal nome della funzione.
- **FlashAttention**: la funzione
  [`launch_fattn` in fattn-common.cuh](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/fattn-common.cuh#L983-L1007)
  impone `Q->type == GGML_TYPE_F32` e
  `KQV->type == GGML_TYPE_F32`, mentre K/V sono
  separatamente BF16 nell'esperimento. Quindi **query
  e output dell'operazione Flash sono F32**, ma ciò
  non rende automaticamente F32 gli operandi dei kernel
  Tensor Core MMA_F16 o tutte le somme/intermedie.
- **BF16 MMA hardware/cuBLAS non significa Flash MMA_BF16**:
  il kernel Flash prefill osservato è ancora
  `FLASH_MMA_F16`; l'uso di `CUBLAS_COMPUTE_32F`
  per le proiezioni BF16 non implementa il percorso
  FlashAttention BF16 nativo mancante dal dispatcher.

- **MatVec BF16/F32 e accumulazione F32**: in
  [mmvf.cu, `mul_mat_vec_f<T,type_acc>`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/mmvf.cu#L7-L15)
  gli argomenti sono `const T *x`, `const float *y`
  e `float *dst`. Nella funzione lo stato
  `float sumf[ncols_dst]` usa una riduzione F32, e
  l'istanza eseguita nel trace è espressamente
  `mul_mat_vec_f<__nv_bfloat16,float,...>`.
  La selezione del percorso per T=BF16 usa
  `type_acc=float`
  [mmvf.cu, dispatcher](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/mmvf.cu#L609-L632).
  Per questa istanza **input pesi BF16, vettore F32,
  riduzione F32 e output F32 sono supportati congiuntamente
  da simbolo osservato e sorgente pinned**; a differenza
  dei nomi CUDA generici, la precisione dell'accumulo
  è dunque verificata per questo specifico template
  (non per tutti i matvec eseguibili).
  `MATVEC_BF16_F32_NAMED` compare **10.060 volte**,
  di cui 10.048 decode e 12 prefill, con
  749,2876 ms di tempo kernel GPU sommato.

Il source audit sposta il livello di certezza per RMSNorm,
RMSNorm/RoPE, Flash Q/output e GEMM/cuBLAS BF16 + compute F32:
il sorgente pinned **dimostra come questi percorsi sono
implementati**. Resta da completare la corrispondenza per ogni
istanza tra operazione GGML, launch GPU e dtype accumulator/output;
in particolare `cublasLt::splitKreduce_kernel`
e le altre operazioni non mappate ai rispettivi operatori GGML
richiedono ancora un'analisi dei template e della semantica
di accumulo. Il template MatVec BF16/F32 sopra citato
costituisce una **verifica sorgente mirata** e non modifica
il campo globale `arithmetic_precision_coverage_complete=false`.

### 9.2 Stato del gate

Questa prova **non** modifica `production_qualified=false`.
Non eseguire nuovi workload GPU per colmare le lacune senza
autorizzazione separata, né inferire benefici percentuali
di un ipotetico Flash MMA_BF16 dal tempo cumulativo
dei kernel di conversione.

## 10. Audit temporale cast F32→BF16 ↔ GEMM BF16 per intervallo NVTX

**Nuovo strumento offline pubblicato:**
`ruth_runtime.gemma3_phase28_cast_gemm_audit`.
Valido intenzionalmente **solo** per il candidato `bf16-on`,
contesto `2048` e un export Nsight **node-level** della fase 28.
Nessuna nuova inferenza, download, patch al kernel o alla produzione.

Motivazione: il primo inventario ha contato
**17.184** conversioni F32→BF16 e **17.184** GEMM BF16 nello
stesso prefill. È una corrispondenza numerica necessaria ma
**non sufficiente** a dimostrare il consumo del buffer convertito.
Il nuovo lettore affronta il passaggio verificabile successivo:

1. Applica gli stessi limiti dei lettori precedenti per il file
   locale `.ruth-build/phase28/*.sqlite` in sola lettura.
2. Usa l'audit esistente del singolo nodo per ottenere l'associazione
   **già provata** da `graphId` a replay `cudaGraphLaunch` e
   fase NVTX dentro **la medesima acquisizione**.
3. Assegna i kernel non-Graph alle chiamate runtime mediante
   `correlationId`; l'intervallo **CPU** completo della API
   deve trovarsi in un **solo** range NVTX sullo stesso thread.
   Non scambia mai timestamp GPU per timestamp CPU.
4. Controlla **tutti** i 96 intervalli di prefill e i 64 di decode
   singolarmente; distingue modalità `graph_node` e `direct`,
   e confronta per intervallo i conteggi F32→BF16 e GEMM BF16,
   con 52 conversioni BF16→F16 attese dai 26 layer.
   Un eventuale numero uniforme di cast per intervallo
   è **ricavato dai dati**, non imposto a 179.
5. Verifica separatamente se la sequenza per timestamp GPU rende
   possibile una corrispondenza ordinata (il cast deve
   terminare **prima** dell'inizio del GEMM candidato).
   Sui replay confronta anche l'ordine degli
   identificatori di nodo osservati. Nessun ordinamento
   dimostra dipendenze di memoria o archi di DAG.
6. Pubblica i conteggi, le distribuzioni e le somme delle durate
   kernel GPU per fase e modalità; una GPU-time sum non è
   un possibile speedup end-to-end.

Comandi:

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_cast_gemm_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_cast_gemm_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048
```

**Test sintetici inclusi:** caso normale con range direct/graph,
conteggi alterati, ordine GPU contrario, nomi riconosciuti; il
successo sul database reale è da confermare dal risultato dell'host.

Interpretazione delle bandiere:

- `per_range_attribution_verified`: ogni kernel target è assegnato
  a un intervallo e l'uguaglianza di conteggio F32→BF16/GEMM
  è rispettata individualmente nel prefill.
- `gpu_timeline_candidate_pairing_verified`: ordinamento
  temporale compatibile con possibili coppie in ogni intervallo.
  È **solo evidenza di compatibilità**, non un join di buffer.
- `graph_node_kernel_order_stable`: lista degli ID di nodo
  osservati nello stesso ordine nelle repliche del grafo
  (non un arco producer→consumer del CUDA Graph).
- `producer_consumer_buffer_dependency_verified=false`,
  `ggml_operator_identity_verified=false` e
  `potential_conversion_elimination_verified=false`
  **sempre**, finché non siano disponibili metadati di memoria,
  dipendenze DAG o tracciamento mirato GGML; non trasformare
  un match temporale in una prova causale.

L'eventuale passo successivo è una cattura strumentata dei
**puntatori/offset degli operandi GGML** e della relazione che
collega ciascun cast al relativo GEMM, oppure un inventario
statico rigoroso dei dati/archi del grafo. Queste estensioni
vanno progettate prima di ogni ulteriore profiling GPU e
richiedono nuova autorizzazione se comportano esecuzioni.

## 11. Esito reale dell'audit cast→GEMM per NVTX e unico outlier GPU

L'utente ha verificato il modulo `gemma3_phase28_cast_gemm_audit`
sull'acquisizione node-level esistente:
**sei test sintetici superati**, `per_range_attribution_verified=true`,
`prefill_per_range_count_balance_verified=true`,
`graph_node_kernel_order_stable=true`,
`unattributed_or_ambiguous_target_kernels=0`,
`target_kernel_records_correlated=39360`.

Per ciascuno dei **96 range prefill**, diretti e in Graph,
è stata confermata la stessa quantità
(**179 cast F32→BF16, 179 GEMM BF16 e 52 cast BF16→F16**).
Nei 64 range decode nessuna di queste famiglie è osservata.

| Fase / percorso | Range | Cast F32→BF16 | GEMM BF16 | Cast BF16→F16 | Range con pairing temporale compatibile |
|---|---:|---:|---:|---:|---:|
| Prefill direct | 28 | 5.012 | 5.012 | 1.456 | **28/28** |
| Prefill Graph | 68 | 12.172 | 12.172 | 3.536 | **67/68** |
| Decode direct | 4 | 0 | 0 | 0 | n/a |
| Decode Graph | 60 | 0 | 0 | 0 | n/a |

**Attenzione:** `gpu_timeline_candidate_pairing_verified=false`.
In uno dei 68 replay prefill, la regola euristica
«cast ordinato per tempo di inizio GPU terminato prima del GEMM
allo stesso indice della sequenza ordinata» non è soddisfatta.
L'outlier **non** invalida l'attribuzione dei 39.360 record,
non implica un problema numerico e non dimostra dipendenza o
indipendenza dei buffer. Anche la lista dei `graphNodeId` ordinati
risulta stabile nei replay (`graph_node_kernel_order_stable=true`),
il che non equivale agli archi del DAG.

Il lettore è stato quindi esteso con diagnostica offline opzionale
per riportare **al massimo quattro range incompatibili** e
**otto coppie discordanti per range**, inclusi:
indice NVTX, correlation ID del replay, indice ordinale,
offset dei timestamp GPU (in nanosecondi, relativi al primo evento
GPU del gruppo), gap, streamId, graphId e graphNodeId.
Non utilizza timestamp CPU come base per i timestamp GPU,
non modifica il modello e non cambia il gate di produzione.

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_cast_gemm_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_cast_gemm_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048 --explain-order
```

L'assenza di archi di dipendenza e riferimenti ai buffer
significa che restano `producer_consumer_buffer_dependency_verified=false`,
`ggml_operator_identity_verified=false` e
`potential_conversion_elimination_verified=false`.
Non ipotizzare accelerazioni percentuali a partire dal solo
tempo di conversione GPU cumulato.

## 12. Outlier localizzato nel replay prefill con correlationId 10072

L'utente ha rieseguito l'audit con `--explain-order`, senza
nuova inferenza. Rimangono veri
`per_range_attribution_verified=true`,
`graph_node_kernel_order_stable=true`,
`prefill_per_range_count_balance_verified=true`,
`unattributed_or_ambiguous_target_kernels=0`. Il risultato
`gpu_timeline_candidate_pairing_verified=false` proviene da
**un unico replay**: range NVTX prefill con indice zero-based
`1`, `graphId=2`, CUDA runtime graph-launch
`correlationId=10072`, con **quattro** coppie ordinali
cast F32→BF16 e GEMM BF16 che hanno una differenza temporale
negativa (fine cast meno inizio GEMM). Tutti i kernel hanno
`streamId=14` in questo specifico replay.

| Indice ordinale | graphNodeId cast | graphNodeId GEMM | start GEMM − end cast (ns) |
|---:|---:|---:|---:|
| 36 | 8589934753 | 8589934754 | −608 |
| 45 | 8589934797 | 8589934798 | −224 |
| 56 | 8589934842 | 8589934843 | −160 |
| 61 | 8589934865 | 8589934866 | −160 |

La distanza massima è 608 ns, dunque **sotto un
microsecondo**. Questa rilevazione da sola **non dimostra**
alcuna violazione del DAG, una dipendenza di memoria, un
errore numerico o un costo di conversione eliminabile.
L'ordine dei kernel in un trace GPU non contiene gli archi
espliciti producer→consumer del grafo né i puntatori dei buffer.
Possibili effetti della granularità temporale o del profiler
sono ipotesi, non conclusioni provate.

### 12.1 Ulteriore diagnostica: stesso graphNodeId in tutti i replay

Il lettore è stato esteso affinché `--explain-order` aggiunga
`same_node_ids_across_replays` a ogni coppia incompatibile.
Per il medesimo `graphId` e per i **medesimi** due
`graphNodeId` osservati, cerca i nodi nelle altre esecuzioni
del prefill già registrate. I nuovi campi riportano:

- `node_pair_replay_count_observed` e
  `node_pair_replays_missing_or_ambiguous`: numero di
  confronti con nodi presenti esattamente una volta;
- `negative_gap_count`, `zero_gap_count`,
  `positive_gap_count`: frequenza del segno del gap;
- `min_gpu_gap_ns`, `median_gpu_gap_ns`,
  `max_gpu_gap_ns`: distribuzione in nanosecondi
  della stessa coppia di ID nei replay;
- `same_stream_id_replay_count` e
  `stream_id_unavailable_replay_count`.

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_cast_gemm_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_cast_gemm_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048 --explain-order
```

In entrambi i casi (anomalia isolata o ricorrente),
`producer_consumer_buffer_dependency_verified=false`,
`ggml_operator_identity_verified=false` e
`potential_conversion_elimination_verified=false`.
Il test sintetico nuovo distingue un replay discordante da
68 replay discordanti; va eseguito sul sistema dell'utente
prima di attribuirgli lo stato «passato» sull'hardware reale.

## 13. Confronto reale degli stessi quattro nodeId sui 68 replay

L'utente ha eseguito `--explain-order` aggiornato sullo **stesso**
export node-level e ha restituito i confronti reali con
`same_node_ids_across_replays`. Per tutte le quattro coppie,
il controllo ha identificato esattamente gli stessi due
`graphNodeId` in **68/68 replay**, senza nodi mancanti o ambigui;
`same_stream_id_replay_count=68`,
`stream_id_unavailable_replay_count=0`.

| Posizione ordinale cast/GEMM | Gap anomalo (ns) | Min (ns) | Mediana (ns) | Max (ns) | Gap negativi / positivi / zero |
|---:|---:|---:|---:|---:|---:|
| 36 | −608 | −608 | +32 | +64 | 1 / 67 / 0 |
| 45 | −224 | −224 | +32 | +64 | 1 / 67 / 0 |
| 56 | −160 | −160 | +32 | +64 | 1 / 67 / 0 |
| 61 | −160 | −160 | +32 | +64 | 1 / 67 / 0 |

Tutti e quattro i gap negativi sono concentrati nello
**stesso replay** (`graphId=2`, `correlationId=10072`,
`nvtx_range_index_zero_based=1`, prefill, stream 14).
Negli **altri 67 replay di ciascuna coppia**, i gap sono
positivi. Per ciascuna coppia gli ID dei nodi risultano
stabili e la mediana delle 68 osservazioni è +32 ns.

Questi risultati **escludono che il gap negativo sia
sistematico nei replay osservati** per questi stessi
identificatori. Sono coerenti con un effetto isolato
di temporizzazione, acquisizione o ordinamento, ma
**non identificano la causa** e non provano che il
tempo negativo sia soltanto rumore del profiler.
Le coppie sono tuttora formate **per posizione ordinale
delle liste ordinate per start GPU**, non tramite
buffer condiviso, `cudaGraphAddDependencies` o un
arco di dipendenza estratto dal DAG. Anche quando
i due kernel appartengono allo stesso stream, i
timestamp esportati non costituiscono da soli una
prova sugli indirizzi di memoria dei loro operandi.

**Decisione tecnica:** non modificare il criterio
`gpu_timeline_candidate_pairing_verified=false`
né introdurre tolleranze arbitrarie da 608 ns o
quantità analoghe per forzarlo a `true`.
Non ripetere nuovi benchmark BF16 ON per inseguire
questo singolo outlier, finché non emerga una
domanda sperimentale ulteriore e specifica.

### 13.1 Prossima verifica rilevante: identità dei buffer

L'evidenza causale utile è il collegamento tra
l'**output BF16** di un lancio `convert_unary`
e l'**operando A/B BF16** del GEMM associato,
possibilmente ricostruendo anche la provenienza
dell'input F32 da un nodo GGML:

1. Nel sorgente pinned il percorso
   `ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>`
   seleziona conversioni `src0/src1` quando
   `type != GGML_TYPE_BF16`, usa buffer temporanei
   `src0_alloc/src1_alloc` e passa `src0_ptr/src1_ptr`
   a cuBLAS. **Questa è una dipendenza di dati
   verificata nel codice del percorso generale**,
   ma non è stata ancora attribuita singolarmente
   a ciascuno dei 17.184 GEMM del trace.
2. Prima ricostruire staticamente le varianti
   `cublasGemmEx`, `cublasGemmStridedBatchedEx`
   e `cublasGemmBatchedEx` e le condizioni di cast:
   src0/src1, `compute_type`, shape, output e
   pool allocazioni. Questo passo non richiede GPU.
3. Verificare se la traccia SQLite contiene **effettivi**
   indirizzi di memoria o relazioni tra nodi.
   Se assenti, non generarli per deduzione da
   ordine/ID/numero di kernel.
4. Se indispensabile, progettare **strumentazione
   sperimentale esplicitamente autorizzata** per
   acquisire le relazioni produttore-consumatore
   sui puntatori dei buffer o sugli archi
   `cudaGraphNode_t`, minimizzando esposizione
   degli indirizzi (identificatori opachi stabili
   nel processo, non puntatori grezzi nei report).
   Nessun nuovo profiling o inferenza è autorizzato
   da questo documento.

Permangono **falsi**:
`producer_consumer_buffer_dependency_verified`,
`ggml_operator_identity_verified`,
`potential_conversion_elimination_verified`,
`full_precision_coverage_verified`,
`production_qualified`.
Lo stato verificato è la **corrispondenza numerica,
temporale e per-run** dell'attività osservata,
non una dimostrazione del dataflow per ciascun GEMM.

## 14. Provenienza dei buffer delle GEMM BF16: audit statico del cuBLAS pinned

**Implementazione:** `ruth_runtime.gemma3_phase28_cublas_source_audit`,
un lettore **offline, di sola lettura e fail-closed** che ispeziona
`.ruth-src/llama-cpp-python/vendor/llama.cpp/ggml/src/ggml-cuda/`
dopo aver eseguito **`source_build_plan.inspect_checkout`**
già presente in Ruth: commit padre
`1652066e0af45f2313b339670ef9555e8a54e545` e commit
submodule llama.cpp
`0c1e57098bba43ac29e6e3b677cdceebdd22334f`,
origin esatto e working tree pulito. Non scarica sorgenti,
non crea un altro clone e non autorizza GPU/compilazione.

### 14.1 Provenienza di src0 e src1 — proprietà provate dal sorgente

Nella funzione
[`ggml_cuda_mul_mat_cublas_impl`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1404-L1491)
il percorso è esplicito:

| Tipo GGML all'ingresso rispetto a `compute_type` selezionato | `src0_ptr` / `src1_ptr` passato a cuBLAS | Nuovo cast GPU? |
|---|---|---|
| Stesso tipo, tipicamente `src0` (peso BF16) | Punta direttamente a `src0->data` oppure `src1->data` | **No**, nel ramo di uguaglianza |
| Tipo diverso, tipicamente `src1` (attivazione F32) | Punta a `src0_alloc.get()` oppure `src1_alloc.get()`, pool temporaneo | **Sì**, tramite `traits::convert` se contiguo, o `traits::convert_nc` altrimenti |
| Tipo diverso su `src0` e `src1` | Entrambi i puntatori possono puntare ad allocazioni temporanee distinte | **Fino a due** cast; non assumere un cast per ogni GEMM universalmente |

Per il `compute_type=GGML_TYPE_BF16`,
[`batched_mul_mat_traits<GGML_TYPE_BF16>`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1376-L1392)
specifica `CUDA_R_16BF`, `nv_bfloat16`,
`CUBLAS_COMPUTE_32F` e lookup `ggml_get_to_bf16_cuda`.
Quest'ultimo, in
[`convert.cu`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/convert.cu#L496-L550),
sceglie `convert_unary_cont_cuda<float>` per sorgente F32.
Il lancio vettoriale
[`convert_unary_cont_vec4`](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/convert.cu#L482-L495)
è **condizionato** a numero di elementi multiplo di
quattro e indirizzi con allineamento sufficiente;
altrimenti è usato il percorso alternativo.
Il simbolo vettoriale è stato osservato nella traccia.

**Interpretazione:** il codice dimostra un legame
**interno al percorso generico** fra buffer prodotto dal cast,
`src0_ptr/src1_ptr` e operando della GEMM cuBLAS.
Il trace ha mostrato **179 cast F32→BF16 e 179 GEMM BF16
per ciascuno dei 96 intervalli di prefill**, ma non ha
associato i **singoli** 17.184 cast a indirizzi o chiamate
GGML specifici. Perciò
`observed_17184_cast_to_gemm_buffer_identity_verified=false`
resta la classificazione corretta.

### 14.2 Varianti cuBLAS, forma dei tensori e precisione

Il percorso
[`ggml-cuda.cu` (branch cuBLAS)](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1517-L1614)
distingue:

| Condizione del sorgente | API cuBLAS invocata | Nota |
|---|---|---|
| `compute_type=F32`, `ne12=ne13=1` | `cublasSgemm` | Ramo **F32**, non BF16 |
| `ne12=ne13=1`, nel ramo successivo | `cublasGemmEx` | BF16 possibile: operandi CUDA BF16, compute F32 |
| Assenza broadcast `r2=r3=1` e layout compatibili | `cublasGemmStridedBatchedEx` | BF16 possibile |
| Altri layout e/o broadcast | `cublasGemmBatchedEx` | Prepara array di puntatori con `k_compute_batched_ptrs` |

Un output GGML `dst` F32 non significa che **tutti**
gli intermedi dell'operazione siano F32.
Per Ada, il BF16 attiva `prefer_f32_output`: il
[puntatore di output destinazione è direttamente F32](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1492-L1523);
negli altri casi i risultati non F32 passano eventualmente
da una [conversione di output](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1609-L1615).
La **configurazione** F32 di cuBLAS è verificata nel sorgente;
le scelte *istruzione per istruzione* dei kernel interni
proprietari cuBLAS/cuBLASLt non sono certificate da questo audit.

Il `compute_type` iniziale deriva dal tipo di `src0`,
ma può cambiare a causa di capacità hardware, richiesta
`GGML_PREC_F32` e variabile
`GGML_CUDA_CUBLAS_COMPUTE_TYPE`
[ggml-cuda.cu:1617–1665](https://github.com/ggml-org/llama.cpp/blob/0c1e57098bba43ac29e6e3b677cdceebdd22334f/ggml/src/ggml-cuda/ggml-cuda.cu#L1617-L1665).
Il dispatcher generale ha anche alternative a cuBLAS,
compreso `mul_mat_vec_f` usato nel decode;
dunque **un modello con pesi BF16 non prova da solo che
ogni proiezione usi cuBLAS BF16**.

### 14.3 Audit offline ripetibile, con gate sorgenti già fissato

Il nuovo comando non avvia CUDA, non modifica file e non
si basa sui soli simboli Nsight. Controlla undici
pattern di struttura del sorgente **entro le funzioni
pertinenti**; non è un parser C++ o una verifica
del dataflow runtime.

**Verifica upstream già eseguita:** sono state lette le due
unità sorgente esattamente al commit
`0c1e57098bba43ac29e6e3b677cdceebdd22334f`
e i medesimi pattern circoscritti sono stati controllati:
**11/11 presenti**. Questo non equivale ad aver eseguito il
modulo Python sul computer dell'utente, né certifica
un checksum della sua copia locale o il suo stato Git.
I quattro test sintetici nuovi sono pubblicati, ma
la loro esecuzione locale non è ancora stata riportata. Fallisce se manca il checkout
approvato, se non corrispondono i commit o se il
checkout è sporco (seguendo l'esistente
`inspect_checkout` di Ruth).

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_cublas_source_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_cublas_source_audit
```

Interpretazione del report:
`source_control_flow_patterns_verified=true` significa
soltanto che il **sorgente del commit approvato** contiene
i rami e le chiamate attese; non prova i parametri
runtime delle 17.184 GEMM. Se la checkout
`.ruth-src/llama-cpp-python` non esiste o è sporca,
il comando restituisce `status=pinned_cublas_source_unverified`
o un errore equivalente senza alterarla.

Le bandiere di dipendenza sperimentale,
`observed_17184_cast_to_gemm_buffer_identity_verified`,
`individual_cuda_graph_buffer_dependencies_verified`,
`ggml_operator_to_kernel_identity_verified`,
`conversion_elimination_safe_verified` e
`production_qualified` restano **false**.

### 14.4 Opzioni successive, senza ottimizzazioni premature

La decisione sull'eliminazione dei cast deve distinguere due
problemi **diversi**:

- Le conversioni **F32→BF16 per le GEMM BF16** preparano
  l'attivazione in un formato accettato dal backend.
  Toglierle senza cambiare altro sarebbe scorretto; possibili
  soluzioni sono riuso/fusione del cast, formato diverso
  degli intermedi oppure un diverso percorso di GEMM,
  verificando qualità e costo del cambiamento.
- Le conversioni **BF16→F16 K/V nel FlashAttention prefill**
  alimentano `FLASH_MMA_F16`. Possono motivare un
  futuro kernel Flash MMA BF16 nativo, ma il suo sviluppo
  è indipendente dalle conversioni cuBLAS e richiede
  verifica della matematica interna e dei logits.

**Prossimo confine sperimentale:** se si vorrà collegare
ogni cast al suo GEMM specifico, identificare in modo
stabile i nodi GGML e i buffer di destinazione, e verificare
l'effettivo matching degli operandi delle API CUDA/cuBLAS
senza mostrare puntatori grezzi nei report. Richiedere
autorizzazione esplicita prima di nuove catture,
esecuzioni GPU o modifiche della libreria pinned.

## 15. Audit sorgente eseguito sul computer dell'utente

L'utente ha eseguito
`.venv/bin/python -m ruth_runtime.gemma3_phase28_cublas_source_audit`
sul proprio checkout Ruth,
restituendo `status=pinned_cublas_source_control_flow_patterns_verified`,
`source_checkout_status=verified`,
`source_checkout_verified=true`,
`source_control_flow_patterns_verified=true` e
`all_pinned_source_patterns_recognized=true`.

Tutte le **11 condizioni** risultano `true`, incluse
`src0_bf16_reuse_or_convert`,
`src1_bf16_reuse_or_convert`,
`bf16_gemm_uses_converted_operand_ptrs`,
`bf16_trait_cublas_f32_compute`,
`bf16_output_f32_on_nvidia`,
`cublas_variants_distinguished`,
`dispatch_precision_overrides` e
`vec4_conversion_conditional_not_universal`.

La verifica Git locale restituisce esattamente
`source_parent_commit_observed=1652066e0af45f2313b339670ef9555e8a54e545`
e `source_llama_cpp_commit_observed=0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
Non è stato necessario scaricare, compilare o eseguire
modelli. Il risultato non sostituisce un'ispezione
AST né prova i buffer di ogni GEMM osservata.
Il risultato riportato non include l'esecuzione
della suite unitaria dei cinque test sintetici nuovi.

**Gate rimasti falsi:**
`observed_17184_cast_to_gemm_buffer_identity_verified`,
`individual_cuda_graph_buffer_dependencies_verified`,
`ggml_operator_to_kernel_identity_verified`,
`source_src0_src1_branch_identity_verified`,
`conversion_elimination_safe_verified`,
`production_qualified`.

### 15.1 Verifica in sola lettura di ciò che l'export Nsight contiene

Prima di progettare una cattura CUDA strumentata,
è stato pubblicato
`ruth_runtime.gemma3_phase28_nsight_schema_audit`,
che riutilizza il gate `inspect_export`:
file SQLite approvato nella cartella Phase28,
dimensioni limitate, tipo Flash atteso `on`,
**nessuna lettura dei valori di puntatori** e
nessuna esecuzione GPU.

Ispeziona **solo i nomi** delle tabelle e delle
colonne SQLite con `sqlite_master` e `PRAGMA table_info`;
segnala separatamente: campi identificativi
`correlationId`, `graphId`, `graphNodeId`,
`streamId`; possibili tabelle di archi DAG;
possibili colonne di puntatori/buffer; e
possibili colonne di parametri delle API.

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_nsight_schema_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_nsight_schema_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048
```

Lo scopo è capire se possiamo ottenere nuove evidenze
**da dati già acquisiti**. Il risultato
`schema_survey_verified=true` significherà
soltanto che i metadati dello schema sono stati
letti e filtrati. La presenza di una colonna
dal nome `devicePointerAddress`, di una
tabella `GRAPH_EDGE` o di campi simili
costituirebbe un candidato da ispezionare,
**non** una verifica degli indirizzi o delle
dipendenze effettive. La loro assenza non
dimostrerebbe invece che sia impossibile
ottenere tali dati con un altro strumento.

I campi
`raw_operand_pointer_values_examined`,
`explicit_graph_edge_data_verified`,
`cublas_operand_pointer_identity_verified`,
`observed_cast_gemm_dependency_verified`,
`ggml_operator_identity_verified`,
`production_qualified`
rimangono tutti `false` per costruzione.
Lo strumento non pubblica indirizzi grezzi.
I test sul checkout dell'utente e il report
dello schema non sono ancora stati eseguiti.

## 16. Esito reale dello schema Nsight e significato di virtualAddress

L'utente ha eseguito sul proprio host
`gemma3_phase28_nsight_schema_audit --candidate bf16-on --n-ctx 2048`
sulla traccia node-level approvata.
Sono risultate **48 tabelle SQLite**,
`schema_survey_verified=true`,
`expected_flash_consistent_with_kernel_names=true` e
`required_activity_tables_present=true`.

Nello schema esistente delle tabelle primarie:

- `CUPTI_ACTIVITY_KIND_KERNEL`: `correlationId`,
  `graphId`, `graphNodeId`, `streamId`, timestamp e
  configurazione dei lanci; **nessun** campo sorgente/destinazione
  degli operandi GEMM;
- `CUPTI_ACTIVITY_KIND_RUNTIME`: `correlationId`,
  `nameId`, timestamp e thread ID; **nessuna**
  colonna candidata per parametri delle API;
- `NVTX_EVENTS`: tempi, tag e payload generici;
  nessuna relazione tra operandi GEMM;
- `explicit_graph_edge_table_candidates=[]`:
  nessuna tabella identificata con archi
  espliciti di dipendenza del DAG.

La sola colonna candidata a indirizzo era
`CUPTI_ACTIVITY_KIND_MEMCPY.virtualAddress`.
Secondo lo **schema SQLite NVIDIA Nsight Systems** è il
**virtual base address delle pagine trasferite**,
non un campo A/B di GEMM e **non** un puntatore
dei parametri passati a cuBLAS. Fonte ufficiale:
[Post-Collection Analysis Guide, schema `CUPTI_ACTIVITY_KIND_MEMCPY`](https://docs.nvidia.com/nsight-systems/AnalysisGuide/#sqlite-schema).
Il campo può aiutare a diagnosticare eventi di trasferimento,
ma non è valido usarlo come identità del buffer prodotto
dal kernel `convert_unary` senza altre evidenze indipendenti.

La presenza di `graphNodeId` e `correlationId`
documenta gli **eventi registrati**, non l'effettiva
dipendenza tra i due kernel. `CUDA_GRAPH_NODE_EVENTS`
(se presente) documenta eventi dei nodi e può includere
`originalGraphNodeId` per nodi clonati; **non
costituisce una tabella di archi DAG**.

### 16.1 Metadati ausiliari, senza lettura degli indirizzi

L'audit è stato esteso per restituire
`auxiliary_event_table_metadata` con **solo presenza,
conteggio delle righe e campi identificativi**
per quattro tabelle, quando presenti:
`CUBLAS_EVENTS`, `CUDA_GRAPH_NODE_EVENTS`,
`CUDA_GRAPH_EVENTS`,
`CUPTI_ACTIVITY_KIND_MEMCPY`.
Non estrae alcun valore di `virtualAddress`,
buffer, payload NVTX o parametri di cuBLAS;
non modifica il database.

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_nsight_schema_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_nsight_schema_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048
```

Questi **sei test** e i nuovi conteggi
non sono ancora stati eseguiti sul checkout
locale dell'utente. Non attribuire loro risultati
prima della verifica.

**Decisione:** la traccia disponibile è sufficiente
per la correlazione NVTX/CUPTI per range/replay,
ma **non** dimostra l'identità A/B di ciascuna
delle 17.184 GEMM BF16. Evitare ulteriori pairing
ordinali di timestamp: un avanzamento richiederà
evidenze sui parametri/indirizzi degli operandi
o sugli archi del DAG, previa autorizzazione
di una strategia sperimentale dedicata.
Rimangono falsi:
`observed_cast_gemm_dependency_verified`,
`cublas_operand_pointer_identity_verified`,
`conversion_elimination_safe_verified`,
`production_qualified`.

## 17. Riferimenti interni

- [Fase 27 — precisione e benchmark](phase-27-gemma3-precision-and-kernel-benchmark.md).
- [Fase 28 — prove CUDA/NVTX e contesti](phase-28-gemma3-dispatch-and-context-scaling.md).
- [Phase 28 gate](../ruth_runtime/gemma3_phase28_gate.py) e [worker](../ruth_runtime/gemma3_phase28_worker.py).
- [Audit dei nodi CUDA Graph](../ruth_runtime/gemma3_phase28_node_audit.py).
- Codice sorgente pinned `ggml-org/llama.cpp@0c1e57098bba43ac29e6e3b677cdceebdd22334f` e sorgenti CUDA collegati nelle sezioni precedenti.
