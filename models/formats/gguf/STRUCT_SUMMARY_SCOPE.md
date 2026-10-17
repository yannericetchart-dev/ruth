# GGUF-STRUCT-SUMMARY-SCOPE-001 — MIG-013

Approvazione: esplicita dell'utente, 17 ottobre 2026. Base main: `2a2ea80eee7b35c9c2d195e6c0de66ac57ee6550`.

## Funzioni autorizzate

- `inspect_structural_summaries(*, project_root: Path | None = None) -> GGUFStructuralReport`: riepiloga solo i GGUF elencati dal catalogo CAT, senza cercare file né modificare dati.
- Per ciascun modello GGUF usare i lettori bounded già integrati GGUF, META, INDEX, BLOCKS e KBLOCKS. Includere header, dichiarazioni tecniche, descrittori, ingombro byte dei tipi noti, offset, allineamento, intervalli/sovrapposizioni già controllati dai lettori esistenti.
- Esiti per modello e aggregati: `structural_checked`, `size_unverified`, `invalid`; `not_configured` quando non ci sono GGUF da esaminare (catalogo assente, vuoto oppure nessun GGUF). Motivi portabili, niente nomi di tensori o riferimenti al filesystem nei report.
- Validità: INDEX/BLOCKS restano `size_unverified` per tipi quantizzati sconosciuti a quei lettori; l'aggregatore può dichiarare `structural_checked` **solo** quando KBLOCKS verifica gli ingombri di ogni tensore supportato e tutte le verifiche preliminari sono coerenti. Se KBLOCKS non può verificare un tipo, l'esito è `size_unverified`, mai falso successo. Errore di un lettore ha precedenza e produce `invalid`.
- Tredicesimo passaggio Ruth `SUMMARY` dopo FIXTURE e comando mirato `gguf-summary`. Nessuna modifica agli esiti storici e alle 12 fasi esistenti. Catalogo assente = skip non bloccante; `invalid`/`size_unverified` = esito nonzero.
- Test con fixture sintetiche di scalare, Q4_0, Q4_K, tipo sconosciuto, header malformato, estensione troncata, catalogo mancante/non valido, report immutabili e CLI. CI e mappe ricorsive, PR lungo GGUF -> Formats -> Models -> main, collaudo personale post-merge.

## Esclusioni e garanzie

Nessuna lettura dei byte dei pesi, dequantizzazione, checksum del payload, download, rete, CUDA, backend, inferenza, installazioni o benchmark. Niente nuovi modelli reali nel repository. I codici di stato sono dichiarazioni **solo strutturali**, non attestati di integrità o compatibilità. DATA-001, REMOTE-001 e MIG-FLOW-001/002 vigenti. Kaggle inattivo.

Stato all'avvio: approved; implementation candidate; CI/integration/manual pending.

## MIG-013 integrazione tecnica completata (17 ottobre 2026)

- PR #78 GGUF, #79 Formats, #80 Models e #81 main integrate; commit applicativo
  `6da03a348407c9caec355238c21c6a122a4b6298`.
- CI di tutte le promozioni **success**: unit test sintetici per file validi,
  quantizzazioni standard e K, tipo sconosciuto, header non valido,
  estensione incompleta, catalogo assente/non valido, CLI e regressioni;
  mappe ricorsive coerenti.
- Ruth default ora esegue **13 fasi**: SUMMARY dopo FIXTURE.
  Comando mirato `gguf-summary`; catalogo opzionale assente =
  `not_configured` non bloccante; anomalie = `invalid` oppure
  `size_unverified`, non certificate come successo.
- Stato: implementation **complete**; verification **passed (CI sintetica)**;
  integration **merged**; **GGUF-STRUCT-SUMMARY-MANUAL-001 pending**.
  Dopo aggiornamento conservativo del checkout personale, eseguire Ruth
  senza argomenti e `python -m unittest models.formats.gguf.tests.test_structural_summary -v`.
  Output completo positivo equivale ad accettazione senza secondo via libera
  (MIG-FLOW-002 rev.2).
- Non sono stati verificati GGUF reali, pesi, integrita' del payload,
  CUDA, runtime o inferenza. DATA-001/REMOTE-001 invariati;
  `local_sync: pending_no_access`. MIG-014 non approvata.

## GGUF-STRUCT-SUMMARY-MANUAL-001 accettata; MIG-013 chiusa (17 ottobre 2026)

- Commit main dell'esecuzione personale `49d3c451b0da86acc99ce33ec6c395f0c27e1ec7`, corrispondente al
  ref GitHub heads/main verificato prima della chiusura.
- Default Ruth: **13 controlli completati** in ordine, inclusi FIXTURE e
  SUMMARY, senza errori. Fixture sintetiche: **passed**, tre file.
- SUMMARY: `not_configured` con `CAT:not_configured` e catalogo personale
  opzionale assente; skip valido, non equivale alla verifica di GGUF reali.
- Suite mirata `test_structural_summary`: **11 test, OK**. Inclusi casi
  sintetici di formati supportati e sconosciuti e input strutturali errati.
- Prova positiva sufficiente come accettazione personale (MIG-FLOW-002
  revisione 2): non occorre una seconda conferma. MIG-013: implementation
  **complete**, verification **passed** (CI + collaudo sintetico personale),
  integration **merged** (PR #78-#81), manual acceptance **passed**.
- Nessun GGUF reale, pesi o integrita' payload verificati; nessuna
  dequantizzazione, installazione, download, CUDA, backend o inferenza.
- DATA-001: pubblicati soltanto SHA, esiti e numero test, non log,
  inventari host, percorsi, credenziali, risultati benchmark o dati reali.
- Checkout locale: `local_sync: verified_on_tested_revision` dalla prova
  personale sul commit indicato; dopo la PR documentale di chiusura,
  `local_sync: pending_docs_fast_forward` finche' non vi e' nuova prova.
- Le precedenti note candidate / manual pending sono **storiche**.
  MIG-014 non approvata: per ogni nuovo scope serve approvazione
  esplicita (MIG-FLOW-001).
