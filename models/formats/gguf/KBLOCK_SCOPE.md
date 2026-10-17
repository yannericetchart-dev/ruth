# GGUF-KBLOCK-SCOPE-001 — sei tipi GGML K-quant

Approvazione esplicita utente: 17 ottobre 2026; base main `1b497172bfad562e6096e88ac8be69ef01a8ced6`.
MIG-011, estensione descriptor-only di GGUF-BLOCK-001.

## Contratto funzionale

- Sola lettura dei GGUF dichiarati dal catalogo CAT-001 locale
  opzionale: nessuna scansione, percorso arbitrario, rete o acquisizione.
- `inspect_kblock_layout(model_id, *, project_root=None) -> GGUFKBlockLayout`
  e `inspect_all_kblock_layouts(*, project_root=None) -> GGUFKBlockReport`.
  Report immutabili: id catalogo, codice status, numero di
  descrittori e conteggio degli ingombri non verificabili.
- Sei GGML type IDs aggiuntivi e verificati da ggml.h / ggml-common.h:
  Q2_K (id 10, 256 elementi, 84 byte),
  Q3_K (id 11, 256, 110),
  Q4_K (id 12, 256, 144),
  Q5_K (id 13, 256, 176),
  Q6_K (id 14, 256, 210),
  Q8_K (id 15, 256, 292).
- Formula: per questi sei tipi il primo asse (dimensione interna)
  deve essere divisibile per 256; ingombro = prodotto delle
  dimensioni / 256 * byte del blocco. Tipi con dimensioni non
  divisibili: `invalid_block_dimension`.
- Riuso del lettore bounded INDEX, dei cinque tipi 32-element
  validati da BLOCKS e dei tipi scalari noti. Per tipi ulteriori
  o futuri `size_unverified`, mai risultato di verifica completa.
- Controlli: numero e unicita' dei nomi, dimensioni positive,
  prodotto limitato, offset relativi all'inizio dei dati,
  allineamento, intervalli entro file, sovrapposizioni dei tipi
  il cui ingombro e' determinabile. Bounds: max 16384 descrittori,
  16384 metadati, 64 MiB di struttura esaminata; max 4 assi,
  64 byte/nome, profondita' array metadata bounded e
  prodotto dimensioni massimo 2^63-1.
- Non modificare semantica delle vecchie API INDEX/BLOCKS:
  tipi K continuano ad essere `size_unverified` nelle due fasi
  precedenti, finche' solo KBLOCKS attiva la verifica K.
- `main.py` senza argomenti: undici fasi
  PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF -> META ->
  INDEX -> BLOCKS -> KBLOCKS. Sottocomando mirato
  `main.py gguf-kblocks`. Catalogo assente, vuoto, senza
  GGUF: skip valido non bloccante.
- Test sintetici: 6 formati, GGUF v2/v3, endian, divisibilita',
  file troppo corto, offset, allineamento, overlap, malformed,
  symlink, tipi sconosciuti, regressioni INDEX/BLOCKS e CLI.
  Collaudo personale post-main: esecuzione default di Ruth e
  suite fixture sintetiche locale, senza modello reale.
- Esclusi peso tensor, lettura dati/dequantizzazione, hash,
  integrity proof, compatibilita' del modello, download,
  Hugging Face, CUDA, inferenza, installazioni e benchmark.
  Kaggle inattivo. DATA-001 e REMOTE-001 invariati.
- Promozioni GitHub gguf -> formats -> Models -> main, CI,
  mappe ricorsive; poi GGUF-KBLOCK-MANUAL-001 (MIG-FLOW-002).
  Prossimo scope separato e da approvare.

Stato iniziale: approvato; codice candidato, verifica e
integrazione in main ancora pendenti.

## Stato tecnico integrato (17 ottobre 2026)

- Implementazione verificata con fixture sintetiche e CI,
  promossa lungo PR #65 GGUF, #66 Formats, #67 Models,
  #68 main `a803e15e872d71a96b7b2de37b13a8964d37d168`.
- Stato: implementation complete, CI verification passed,
  integration merged. **GGUF-KBLOCK-MANUAL-001 pending**,
  richiesta esecuzione personale del default Ruth piu'
  fixture sintetiche locali postmerge, accettazione esplicita.
- Nessuna lettura di pesi, dequantizzazione o modello reale
  convalidato; le note candidate precedenti sono storiche.
