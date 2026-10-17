# GGUF-FIXTURE-SUMMARY-SCOPE-001 — MIG-015

Perimetro approvato esplicitamente dall'utente il 17 ottobre 2026. Base main `0cb12ad90e2ef7d1b3b3b77009f9cf476a3e5289`.

- Estendere la dodicesima fase Ruth FIXTURE gia' integrata: la catena sintetica CAT -> GGUF -> META -> INDEX -> BLOCKS -> KBLOCKS deve includere infine **SUMMARY**, eseguita sullo stesso catalogo temporaneo di tre GGUF deterministici senza modelli reali.
- Verificare che SUMMARY sia `structural_checked` su tutti e tre i modelli sintetici (`scalar`, `q4`, `kquant`), con conteggi e motivi esatti e senza reinterpretare gli stati parziali legacy INDEX/BLOCKS. Un esito anomalo deve far fallire il collaudo integrato.
- Test negativi temporanei: tipo GGML ignoto -> `size_unverified`; magic invalida o estensione troncata -> `invalid`; cambiamento durante lettura -> `STABILITY:changed_during_inspection`; il guard del collaudo deve respingere anomalie SUMMARY.
- Ruth conserva **13 fasi**: FIXTURE ora controlla anche SUMMARY, seguita dalla normale fase SUMMARY del catalogo personale; `gguf-fixture` resta il sottocomando mirato, nessuna quattordicesima fase.
- Test mirati `models.formats.gguf.tests.test_fixture_summary` (7 test), regressione `test_pipeline` aggiornata, CI e mappe ricorsive. Promuovere via PR GGUF -> Formats -> Models -> main, poi richiedere collaudo personale default Ruth e test mirati. Output positivo = accettazione senza nuova approvazione (MIG-FLOW-002 rev.2).
- Solo strutture sintetiche in TemporaryDirectory, nessun file reale/modello/peso, download, rete, inferenza, CUDA, dequantizzazione o checksum payload. Controllo di stabilita' best effort resta non atomico; DATA-001, REMOTE-001 invariati; Kaggle inattivo.

Stato iniziale: **approved**, implementazione candidate, CI/integration/personal acceptance pending.

## MIG-015 integrazione tecnica completata (17 ottobre 2026)

- PR #90 GGUF, #91 Formats, #92 Models, #93 main: integrate dopo GitHub
  CI **success** in ogni promozione (test sintetici, regressione `test_pipeline`,
  punto d'ingresso e mappe). Commit applicativo main `91ad6f4e927c0ed2dd96b147a44c074ccbc5bfe4`.
- La dodicesima fase FIXTURE include ora il controllo SUMMARY sullo stesso
  catalogo temporaneo di **3 GGUF sintetici**, richiedendo
  `SUMMARY=structural_checked` e tre rapporti `structural_checked`.
  Il report deve fallire se SUMMARY è anomala. La SUMMARY standard finale
  resta la **tredicesima fase Ruth** sul catalogo locale opzionale, invariata.
- Sette test mirati `test_fixture_summary` superati in CI; coperti
  casi stabili, quantizzazioni note, tipo ignoto, magic non valida,
  extent troncato, modifiche durante la lettura e failure guard.
- Stato: implementazione **complete**, verification **passed (CI)**,
  integrazione **merged**; collaudo personale
  `GGUF-FIXTURE-SUMMARY-MANUAL-001` **pending**.
  Dopo il merge del checkpoint: eseguire Ruth 13 fasi e la suite mirata
  `python -m unittest models.formats.gguf.tests.test_fixture_summary -v`
  (sette test). Evidenza positiva completa vale come accettazione senza
  secondo consenso ai sensi di MIG-FLOW-002 rev.2.
- Non verificati GGUF reali o pesi; nessuna lettura tensor_data, hash
  payload, download, rete, CUDA o inferenza. DATA-001 e REMOTE-001 vigenti.
  `local_sync: pending_no_access`. Nuova migrazione non autorizzata.

## GGUF-FIXTURE-SUMMARY-MANUAL-001 accettata; MIG-015 chiusa (17 ottobre 2026)

- Utente ha fornito prova personale sul commit main `86d7aa5b2000e07ef55798cab3779bf53e3cbf72`,
  identico al ref GitHub `heads/main` verificato prima della chiusura.
- Ruth `main.py` senza argomenti: **13 fasi completate** in ordine, senza
  anomalie; collaudo FIXTURE: **passed** su tre GGUF sintetici,
  `SUMMARY=structural_checked`; INDEX e BLOCKS `partial` previsti per
  quantizzazione legittima, KBLOCKS `inspected`.
- SUMMARY successiva sul catalogo locale personale opzionale:
  `not_configured` / `CAT:not_configured`; skip valido, non collaudo reale.
- Suite mirata `test_fixture_summary`: **7 test, OK**, nessun fallimento;
  comprende casi sintetici struttura valida, tipo sconosciuto, magic errata,
  intervallo troncato, mutazione osservabile e guard di fallimento SUMMARY.
- MIG-FLOW-002 revisione 2: prova positiva = accettazione personale senza
  nuovo consenso. MIG-015: implementation **complete**, verification
  **passed** (CI e test sintetici personali), integration **merged**
  (PR #90-#93), manual acceptance **passed**.
- Nessun GGUF reale, peso, tensor_data o compatibilita' runtime testati;
  nessun download, rete, CUDA o inferenza. DATA-001: nessun log o inventario
  di host, percorso privato, credenziale o benchmark scritto su Git.
- Checkout locale: `local_sync: verified_on_tested_revision` limitatamente
  allo SHA del collaudo fornito; dopo merge dei documenti di chiusura
  `local_sync: pending_docs_fast_forward`. Stato del checkout pulito non
  dimostrato dall'output incollato.
- Le note precedenti `manual pending` sono storiche, non stato corrente;
  MIG-016 **non approvata**. Nuova porzione solo dopo approvazione esplicita
  del perimetro proposto, secondo MIG-FLOW-001.
