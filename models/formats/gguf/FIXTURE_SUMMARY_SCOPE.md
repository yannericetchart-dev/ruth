# GGUF-FIXTURE-SUMMARY-SCOPE-001 — MIG-015

Perimetro approvato esplicitamente dall'utente il 17 ottobre 2026. Base main `0cb12ad90e2ef7d1b3b3b77009f9cf476a3e5289`.

- Estendere la dodicesima fase Ruth FIXTURE gia' integrata: la catena sintetica CAT -> GGUF -> META -> INDEX -> BLOCKS -> KBLOCKS deve includere infine **SUMMARY**, eseguita sullo stesso catalogo temporaneo di tre GGUF deterministici senza modelli reali.
- Verificare che SUMMARY sia `structural_checked` su tutti e tre i modelli sintetici (`scalar`, `q4`, `kquant`), con conteggi e motivi esatti e senza reinterpretare gli stati parziali legacy INDEX/BLOCKS. Un esito anomalo deve far fallire il collaudo integrato.
- Test negativi temporanei: tipo GGML ignoto -> `size_unverified`; magic invalida o estensione troncata -> `invalid`; cambiamento durante lettura -> `STABILITY:changed_during_inspection`; il guard del collaudo deve respingere anomalie SUMMARY.
- Ruth conserva **13 fasi**: FIXTURE ora controlla anche SUMMARY, seguita dalla normale fase SUMMARY del catalogo personale; `gguf-fixture` resta il sottocomando mirato, nessuna quattordicesima fase.
- Test mirati `models.formats.gguf.tests.test_fixture_summary` (7 test), regressione `test_pipeline` aggiornata, CI e mappe ricorsive. Promuovere via PR GGUF -> Formats -> Models -> main, poi richiedere collaudo personale default Ruth e test mirati. Output positivo = accettazione senza nuova approvazione (MIG-FLOW-002 rev.2).
- Solo strutture sintetiche in TemporaryDirectory, nessun file reale/modello/peso, download, rete, inferenza, CUDA, dequantizzazione o checksum payload. Controllo di stabilita' best effort resta non atomico; DATA-001, REMOTE-001 invariati; Kaggle inattivo.

Stato iniziale: **approved**, implementazione candidate, CI/integration/personal acceptance pending.
