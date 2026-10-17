# GGUF-FIXTURE-SCOPE-001 (MIG-012)

Approved 17 ottobre 2026. Three disposable synthetic GGUF fixtures: scalar v2 little-endian, Q4_0 v3 big-endian, Q4_K v3 little-endian, declared only in a temporary CAT manifest. Read-only, bounded existing parsers; 12th Ruth default stage FIXTURE plus `gguf-fixture` targeted command. Negative malformed magic and truncated payload extent tests. Legacy INDEX and BLOCKS retain size_unverified. No real models, weights, downloads, network, inference or CUDA. Unit/CI and recursive maps required; after integration personal GGUF-FIXTURE-MANUAL-001 required (MIG-FLOW-002-R2). DATA-001 and REMOTE-001 unchanged.

Status iniziale storico: approved, candidate. Stato corrente nella sezione di chiusura seguente.

## MIG-012 integrazione tecnica verificata (17 ottobre 2026)

- PR #71 (GGUF), #72 (Formats), #73 (Models) e #74 (main) integrate, commit applicativo main `b057957b5d96b3c09ecde0b6c28ffc0e977c6942`.
- CI per tutte e quattro le promozioni success, inclusi test sintetici, punto d'ingresso e mappe. Dodicesima fase Ruth FIXTURE, dopo KBLOCKS, e comando mirato `gguf-fixture`.
- Stato: implementation complete, verification passed (fixture e CI), integration merged. **GGUF-FIXTURE-MANUAL-001 pending**: richiede collaudo personale del default Ruth e test mirati dopo aggiornamento del checkout personale.
- Non sono stati testati modelli reali o letti pesi. Nessun download, rete, CUDA, backend o inferenza. Nuovo perimetro bloccato fino a collaudo personale e nuova approvazione (MIG-FLOW-001/002). DATA-001 e REMOTE-001 invariati.

## GGUF-FIXTURE-MANUAL-001 accettata; MIG-012 chiusa (17 ottobre 2026)

- L'utente ha fornito la prova personale del commit main `a3ced17b0e4bccaf8fe8e10db9c1f5b5c508bda3`, verificato
  identico al ref GitHub heads/main prima della chiusura.
- L'ingresso Ruth senza argomenti ha completato tutte le dodici fasi in ordine,
  fino a FIXTURE, con riepilogo conclusivo regolare e nessun errore.
- FIXTURE: **passed** su tre file GGUF sintetici; CAT=ready, GGUF=inspected,
  META=inspected, INDEX=partial e BLOCKS=partial attesi per i tipi non
  verificati dalle fasi legacy, KBLOCKS=inspected.
- Suite mirata `test_pipeline`: **4 test, OK**, senza failure.
  I controlli CAT/GGUF/META/INDEX/BLOCKS/KBLOCKS sull'eventuale catalogo
  personale assente erano `not_configured`, comportamento valido.
- Sotto MIG-FLOW-002 revisione 2 la prova completa costituisce accettazione
  personale sufficiente: nessuna seconda autorizzazione richiesta.
  MIG-012: implementation **complete**, verification **passed** (CI e prova
  sintetica), integration **merged** (PR #71-#74), manual acceptance **passed**.
- Nessun GGUF reale o payload di tensori ispezionato nel collaudo personale;
  nessuna validazione di modelli, inferenza, runtime o CUDA.
- DATA-001: non pubblicare output grezzi, path, inventari hardware,
  identificativi host, versioni di pacchetti o benchmark. Solo commit,
  esiti sintetici e numero test. `local_sync: verified_on_tested_revision`
  per lo SHA sopra, secondo prova utente; dopo documenti di chiusura
  `local_sync: pending_docs_fast_forward`.
- Nessuna MIG-013 approvata: nuovo perimetro funzionale richiede verde
  esplicito ai sensi di MIG-FLOW-001. Le note storiche `pending`
  non descrivono lo stato attuale.
