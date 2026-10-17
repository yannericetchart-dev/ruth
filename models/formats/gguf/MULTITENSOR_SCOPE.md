# GGUF-MULTITENSOR-FIXTURE-SCOPE-001 — MIG-016

Approvata espressamente dall'utente il 17 ottobre 2026; base canonica main `3cd6cad3e431f5ad1ab8dc698aee96330ff79ed1`.

- Estendere FIXTURE con **quattro** file sintetici temporanei (i precedenti tre piu' `multi`) in catalogo CAT isolato in TemporaryDirectory. Il quarto GGUF v3 little-endian dichiara **tre tensori** con tipi scalar F32, Q4_0 e Q4_K e intervalli distinti. Pesi assenti: i byte placeholder sono soltanto zeri sintetici e non vengono letti dai controlli.
- Eseguire CAT -> GGUF -> META -> INDEX -> BLOCKS -> KBLOCKS -> SUMMARY sul catalogo sintetico. Attesi INDEX `partial`, BLOCKS `partial`, KBLOCKS `inspected`, SUMMARY `structural_checked` per tutti e quattro i modelli e `tensor_count=3` per `multi`. Nessun `size_unverified` residuo dopo KBLOCKS quando tutti i tipi sono noti.
- Verifiche mirate in `test_multitensor` (11 test): intervalli validi e descrittori fuori ordine, endian big e allineamento personalizzato, offset non allineato, sovrapposizioni, offset fuori file, estensione troncata, offset duplicato, tipo GGML sconosciuto, dimensione blocco non multipla. Mantenere test sintetici MIG-012/013/014/015 aggiornati a quattro fixture.
- Gli esiti negativi devono essere `invalid` con codice portabile esistente o `size_unverified` per tipi sconosciuti. Niente falsa certificazione `structural_checked`. Nessun cambiamento al parser bounded approvato o alla semantica di validazione: collaudo dei controlli esistenti, non introduzione di nuove categorie di quantizzazione.
- Conservare le **13 fasi** di `main.py`, FIXTURE continua a precedere SUMMARY del catalogo personale opzionale, e il sottocomando mirato `gguf-fixture` resta disponibile.
- Seguire REMOTE-001: PR e CI per GGUF -> Formats -> Models -> main, mappe ricorsive e checkpoint; poi test personale di Ruth default e `python -m unittest models.formats.gguf.tests.test_multitensor -v`. MIG-FLOW-002-R2: evidenza positiva completa implica accettazione senza altra domanda.
- DATA-001: nessun log, host inventory, percorsi personali, dati reali, credenziali o benchmark in Git. Nessun modello reale, download, lettura dei pesi, CUDA, rete o inferenza. Kaggle inattivo.

Stato iniziale: **approved**; implementation candidate; CI, integration and manual acceptance pending.

## MIG-016 integrazione tecnica completata (17 ottobre 2026)

- PR #96 GGUF, #97 Formats, #98 Models e #99 main integrate dopo
  GitHub CI **success** a ogni promozione (suite sintetiche, regressioni,
  punto d'ingresso e mappe); commit applicativo main `4eb92225e3addbc8b7d58b15243557dae86e4479`.
- FIXTURE ora dichiara **4 GGUF sintetici** in catalogo temporaneo:
  `scalar`, `q4`, `multi` e `kquant`. `multi` ha tre descrittori
  (F32, Q4_0, Q4_K), offset 0/32/64 e intervalli validi.
  CAT/GGUF/META/INDEX/BLOCKS/KBLOCKS/SUMMARY sempre in sequenza;
  INDEX/BLOCKS possono ancora produrre `partial` per tipi legittimi,
  KBLOCKS completa il dimensionamento, SUMMARY deve essere
  `structural_checked` con `tensor_count=3` per multi.
- Suite mirata `test_multitensor`: **11 test sintetici** passati in CI,
  incluso big endian, allineamento personalizzato, ordine descrittori,
  offset/sovrapposizione, extent insufficiente, tipo sconosciuto.
  Suite legacy fixture adattate da 3 a 4 modelli, senza modifica ai parser.
- Il flusso Ruth resta a **13 fasi**. Stato MIG-016:
  implementation **complete**, verification **passed (CI)**,
  integration **merged**, personal `GGUF-MULTITENSOR-MANUAL-001`
  **pending**. Per collaudo personale: default Ruth e suite
  `python -m unittest models.formats.gguf.tests.test_multitensor -v`
  (11 test). Se positivo su commit main verificato, MIG-FLOW-002-R2
  autorizza la chiusura automatica senza richiesta di ulteriore consenso.
- Nessun GGUF reale/pesi, tensor_data, hash, dequantizzazione, rete,
  download, CUDA o inferenza. DATA-001/REMOTE-001 invariati, Kaggle inattivo.
  `local_sync: pending_no_access`; nessun perimetro MIG-017 approvato.

## MIG-016 chiusa con collaudo delegato (17 ottobre 2026)

- Evidenza portabile: MIG-016-delegated-001; mode: delegated; status: passed.
  Commit main eseguito: dbdef7ab000d99d2e3dde68b0857840f0047d5fe, risolto via
  heads/main e verificato prima del collaudo. Nessuna esecuzione personale
  dell'utente viene attribuita a questa prova.
- Destinazione: nuova copia locale dedicata designata dall'utente, creata
  esclusivamente dal repository canonico; integrita' degli oggetti Git,
  branch main, revisione e pulizia verificati. Prova minima di scrittura,
  lettura e rimozione superata; ambiente venv dedicato e isolato disponibile
  senza installazioni di sistema o pacchetti aggiuntivi.
- Default Ruth: 13 fasi completate nell'ordine prescritto e riepilogo
  finale positivo, exit code 0. FIXTURE: passed su quattro file sintetici,
  SUMMARY=structural_checked. Catalogo opzionale e SUMMARY successiva
  not_configured: skip previsto, non verifica di modelli reali.
- Suite prescritta models.formats.gguf.tests.test_multitensor:
  11 test, OK, nessun fallimento. La suite verifica anche i tre descrittori
  del modello sintetico multi e i casi negativi previsti dallo scope.
- MIG-FLOW-002 revisione 3: queste prove complete soddisfano il gate
  GGUF-MULTITENSOR-MANUAL-001, mantenendone l'ID storico ma registrando
  mode: delegated. Nessuna valutazione umana aggiuntiva era prescritta;
  nessuna ripetizione personale o seconda conferma necessaria.
- Stato MIG-016: implementation complete; verification passed (CI e
  collaudo locale sintetico delegato); integration merged (PR #96-#99);
  local validation passed; closure complete.
- validation_checkout: verified_on_tested_revision. Dopo la pubblicazione
  di questi soli documenti, pending_docs_fast_forward fino alla sync
  conservativa della nuova destinazione; il collaudo dello SHA sopra
  resta valido solo se il diff successivo e' esclusivamente documentale.
- I checkout precedenti restano separati e invariati: copia pulita
  precedente pending_permissions; copia storica modificata preservata
  ed esclusa dalla sync. Non sono dichiarati sincronizzati.
- DATA-001: output grezzi, cache e altre evidenze reali conservati solo
  localmente in spazi autorizzati ed esclusi da Git. Pubblicati soltanto
  SHA, ID, stati e conteggi. Nessun GGUF reale, peso o tensor_data,
  hash payload, compatibilita' runtime, CUDA, download o inferenza verificati.
- MIG-017 non approvata. Dopo questa chiusura presentare il prossimo
  perimetro funzionale e attendere un nuovo OK prima di implementarlo.
  Le note precedenti di collaudo pending sono cronologia storica.
