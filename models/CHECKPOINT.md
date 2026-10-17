# Models: checkpoint CAT-001

- CAT-SCOPE-001 approvato 16 ottobre 2026; solo catalogo locale
  read-only dichiarativo e opzionale, sesto passaggio di Ruth.
- models/catalog implementata, verificata e integrata in main tramite
  PR #36-#38; collaudo manuale CAT-MANUAL-001 superato. Il resto di Models e' ancora scaffold inattivo:
  GGUF, sorgenti/acquisizione Hugging Face, cataloghi remoti,
  verifica degli artefatti e inferenza esclusi.
- CAT-MANUAL-001 completato positivamente dall'utente il 16 ottobre 2026
  sul commit `90e57f106fb664b0257775260044125ea85228e2`.

## CAT-001 verificata e integrata (16 ottobre 2026)

- PR #36 foglia, #37 models/main e #38 main integrate. GitHub CI
  38928909391, 38929028829 e 38929118836 success (116 test sintetici,
  264 mappe). Si tratta soltanto del catalogo locale dichiarativo.
- Il dominio Models resta parziale: catalogo completo per il solo
  perimetro approvato, GGUF/download/Hugging Face non implementati.
- CAT-MANUAL-001 complete: conferma personale successiva alla CI
  e all'integrazione, registrata senza dati host su Git.

## Accettazione manuale CAT-001 (16 ottobre 2026)

- Conferma positiva dell'utente: Ruth eseguita dall'ingresso ufficiale
  sul commit `90e57f106fb664b0257775260044125ea85228e2` (sei controlli in sequenza).
- Catalogo non configurato: `not_configured` valido; non e' richiesta
  la creazione di un manifest per il collaudo.
- Prossima migrazione Models non autorizzata: attendere descrizione
  funzionale e nuova approvazione. DATA-001 e REMOTE-001 invariati.

## GGUF-SCOPE-001 avviato (16 ottobre 2026)

- Dopo CAT-MANUAL-001 e PR #40, perimetro GGUF header-only approvato.
- Lettore in `models/formats/gguf` e settimo passaggio root candidati;
  nessuna integrazione GGUF in main finche' non si concludono promozioni.
- Le restanti porzioni Models (GGUF payload, acquisizione, source Hugging Face)
  rimangono inattive; GGUF-MANUAL-001 dopo merge.

## GGUF-001 verificata e integrata (16 ottobre 2026)

- Reader e orchestrazione approvati (GGUF-SCOPE-001), test sintetici e
  CI passati nei quattro livelli; PR #41-#44 integrate fino a main
  al commit `15b106082c4dee37622c61a8afefd248a57464a0`.
- Il dominio Models resta partial: lettura di intestazione GGUF completata
  soltanto per il perimetro autorizzato, resto skeleton/inattivo.
- GGUF-MANUAL-001 pending; il collaudo personale avverra' dal flusso
  completo di Ruth dopo sincronizzazione sicura del checkout.
  Nessuna migrazione successiva prima della conferma.
- Dati macchina/manifest e pesi del modello rimangono esclusi da Git.

## GGUF-MANUAL-001: accettazione (16 ottobre 2026)

- Conferma positiva esplicita dell'utente dopo esecuzione personale
  dell'ingresso Ruth senza argomenti sul main `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`.
- Le sette fasi si sono concluse regolarmente; GGUF e catalogo
  non configurati costituiscono skip valido, non lettura di GGUF reale.
- Stato del perimetro GGUF-001: complete, verification passed
  (fixture sintetiche), integration merged, manual acceptance passed.
  Le precedenti voci pendenti o candidate sono storiche.
- Nessuna nuova migrazione autorizzata; dati reali esclusi (DATA-001).

## GGUF-META-001 candidata (16 ottobre 2026)

- Perimetro GGUF-META-SCOPE-001 approvato; nuova API bounded,
  ottava fase Ruth, fixture sintetiche e documentazione in sviluppo.
- Implementazione candidata, verifiche e promozioni pendenti;
  nessuna modifica all'accettazione GGUF-MANUAL-001 precedente.
- GGUF-META-MANUAL-001 prevista solo dopo merge in main.
  Download, tensor payload, inferenza e dati locali restano esclusi.

## GGUF-META-001 verificata e integrata (16 ottobre 2026)

- Reader bounded metadata approvato e integrato tramite PR #47-#50,
  main `a630c83f6c62a229e18fe5e64f0863c19de16018`; CI sintetica e mappe verificate in tutti
  e quattro i livelli.
- Solo `general.architecture`, `general.file_type` e
  `{architecture}.context_length` opzionali. Non legge tensori,
  non certifica quantizzazione o compatibilita' inference.
- Stato tecnico del nuovo scope: implementation complete;
  verification passed (fixture sintetiche); integration merged.
- **GGUF-META-MANUAL-001 active/pending**: prova manuale dopo
  sync sicura e avvio default Ruth con tutte le otto fasi.
  Gli stati candidati precedenti sono storici.
- Dati reali esclusi da Git (DATA-001); prossimo scope da approvare.

## GGUF-META-MANUAL-001: accettazione positiva (16 ottobre 2026)

- L'utente conferma esplicitamente positivo il collaudo personale
  del flusso ordinario di Ruth a otto fasi sul main `d318fd793da514f21ee2e8475304d1a6c883ee94`.
- Catalogo, GGUF e META `not_configured`: skip valido.
  Nessuna prova manuale con modello reale ne' certificazione inferenza.
- GGUF-META-001: implementata, CI verificata, integrata (PR #50),
  manual acceptance **passed/complete**. Le note pending precedenti
  restano storiche, non rappresentano lo stato corrente.
- Prossima porzione senza approvazione: richiede nuovo scope
  secondo MIG-FLOW-001/002. DATA-001 e REMOTE-001 invariati.

## GGUF-INDEX-001 candidata (16 ottobre 2026)

- Perimetro GGUF-INDEX-SCOPE-001 approvato. Indici descriptor-only
  da catalogo, nono passo Ruth; implementazione candidata sul
  branch remoto, verifiche/integrazione non ancora eseguite.
- Nessun tensore o dato host nel repo; fixture sintetiche e CI
  ancora da verificare. GGUF-INDEX-MANUAL-001 dopo merge main.
- GGUF-META-MANUAL-001 resta accettata: stato storico precedente
  non deve essere invertito. MIG-FLOW-001/002 e DATA-001 vigenti.

## GGUF-INDEX-001 verificata e integrata (16 ottobre 2026)

- Indice tensori GGUF catalog-bound bounded, nona fase Ruth,
  promosso con PR #53-#56 sino al main `1d51cfeed2bec83119ebf56ff0cb3840089484cf`.
- CI runs 38945221302, 38945375637, 38945463381,
  38945568401 success; verifica sintetica e mappe confermate.
- Stato tecnico: implementation complete, verification passed
  (fixture sintetiche), integration merged.
- **GGUF-INDEX-MANUAL-001 active/pending**, dopo sync main:
  richiesta conferma esplicita dell'utente. Il solo indice non
  certifica modello, payload, quantizzazione o inferenza.
- Non aggiungere real host data a Git (DATA-001); perimetro
  successivo ancora non autorizzato. Note candidate storiche.

## GGUF-INDEX-MANUAL-001: accettazione (17 ottobre 2026)

- GGUF-INDEX-MANUAL-001: **complete/passed**, accettazione esplicita
  dell'utente il 17 ottobre 2026 dopo esecuzione personale del flusso
  ordinario `.venv/bin/python main.py` sul commit `09045243e944514d40b9e58c6159b013efa2a196`,
  identico a main verificato su GitHub.
- Nove fasi in ordine PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT ->
  GGUF -> META -> INDEX; riepilogo finale di completamento regolare.
- Catalogo e letture GGUF/META/INDEX `not_configured` sono skip validi
  e non bloccanti. Nessun file GGUF reale ispezionato personalmente;
  non risultano validati contenuti dei tensori, pesi o compatibilita'
  con CUDA/backend/inferenza.
- La prova manuale completa il gate MIG-FLOW-002 per MIG-009;
  una nuova migrazione richiede uno scope separato e approvazione
  esplicita (MIG-FLOW-001). Nessun nuovo componente avviato.
- DATA-001 e REMOTE-001 invariati: in Git solo esito e commit;
  niente path, nomi modelli, output grezzi, inventari, log o benchmark.
- GGUF-INDEX-001 implemented, CI verified, merged;
  collaudo personale accepted. Le vecchie note di gate pending
  sono storiche. Nessun'altra porzione Models autorizzata.

## GGUF-BLOCK-001 candidata (17 ottobre 2026)

- MIG-010 approvata: verifica ingombro GGML Q4_0, Q4_1,
  Q5_0, Q5_1, Q8_0 senza leggere tensor_data, decimo
  controllo Ruth dopo INDEX. Reader/index invariato per
  le nove fasi gia' integrate e accettate.
- Scope nel GGUF/BLOCK_SCOPE.md, codice candidato remoto:
  test/CI e promozioni ancora pendenti.
  GGUF-BLOCK-MANUAL-001 dopo main; DATA-001 invariato.

## GGUF-BLOCK-001 verificata e integrata (17 ottobre 2026)

- Cinque tipi GGML supportati senza accesso al tensor_data:
  Q4_0/Q4_1/Q5_0/Q5_1/Q8_0. Report immutabili, decima
  fase Ruth; unknown `size_unverified`.
- CI runs 39022476943, 39022574327, 39022637716,
  39022698747 success (fixture, regressioni e mappe).
  PR #59-#62 promosse in main `5f896f1bbacaa8c802f66f2a40ee7663cfa40604`.
- Stato tecnico: implementation complete, verification passed
  (sintetica), integration merged. Manual
  **GGUF-BLOCK-MANUAL-001 active/pending**.
- L'accettazione personale delle nove fasi precedenti resta valida.
  Nessun download, payload, inferenza o dato host in Git (DATA-001).

## GGUF-BLOCK-MANUAL-001 superata (17 ottobre 2026)

- L'utente ha eseguito personalmente il flusso di dieci controlli
  di `.venv/bin/python main.py` sulla revisione `2f42a9d7a631e8779613784b58f8ce1b92afff9c`,
  coincidente con il main pubblicato; ha confermato esplicitamente
  positivo il collaudo GGUF-BLOCK-MANUAL-001 il 17 ottobre 2026.
- Tutte le fasi PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF ->
  META -> INDEX -> BLOCKS sono state completate in ordine.
  Il catalogo opzionale assente e' risultato `not_configured`
  e GGUF/META/INDEX/BLOCKS sono skip validi e non bloccanti.
- Non e' stato collaudato personalmente alcun file GGUF reale:
  l'esito non certifica pesi, integrita', dequantizzazione,
  CUDA, backend, runtime o inferenza.
- I risultati di CI e la verifica con fixture sintetiche dei cinque
  tipi Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 restano distinti
  dall'accettazione manuale senza catalogo.
- DATA-001 e REMOTE-001 rispettati: non vengono versionati
  inventari dell'host, percorsi reali, log, manifest, modelli,
  benchmark o credenziali. Nessuna nuova migrazione approvata.
- Scope GGUF-BLOCK-001 implementato, verificato con fixture in CI
  e integrato in main; **manual acceptance passed**.
  La prossima porzione richiede approvazione distinta.

## GGUF-KBLOCK-001 candidata (17 ottobre 2026)

- MIG-011 approvata; supporto descriptor-only Q2_K, Q3_K,
  Q4_K, Q5_K, Q6_K, Q8_K (blocco 256) nella nuova KBLOCKS.
- Codice candidato in branch remoto; CI, PR e mappe
  non ancora completate. Le 10 fasi precedenti rimangono
  accettate; nessun payload dei tensori letto.
- GGUF-KBLOCK-MANUAL-001 richiede prova personale default
  e fixture sintetica locale dopo merge in main; DATA-001 vigente.

## GGUF-KBLOCK-001 integrata, collaudo pending (17 ottobre 2026)

- Supporto bounded dei sei tipi K-quant con superblocchi 256,
  nessuna lettura pesi; main `a803e15e872d71a96b7b2de37b13a8964d37d168` dopo PR #65-#68.
- CI runs 39025028855, 39025109274, 39025159357,
  39025210189 success: fixture sintetiche e mappe.
- Stato tecnico: implementation complete; verification passed;
  integration merged. **GGUF-KBLOCK-MANUAL-001 active/pending**.
- Test manuale richiesto: Ruth default 11 passi e suite
  test_kblocks.py con fixture sintetiche locale, poi
  conferma esplicita. Vecchie 10 fasi restano accettate.
  DATA-001 e MIG-FLOW-002 invariati.

## GGUF-KBLOCK-MANUAL-001: accettazione personale (17 ottobre 2026)

- Prova comunicata dall'utente il 17 ottobre 2026 per main
  `233efc358f56a66d2abf796fa8e20a476aa788cf`, risolto e confrontato al ref heads/main GitHub:
  flusso normale `.venv/bin/python main.py` concluso senza errori
  su undici fasi in ordine, inclusa KBLOCKS.
- CAT, GGUF, META, INDEX, BLOCKS, KBLOCKS: `not_configured`
  come skip validi con catalogo opzionale assente.
- Suite mirata `test_kblocks.py`: **9 test superati, esito OK**,
  senza failure; verifica manuale sintetica distinta dalla
  verifica precedente CI e dall'uso di un GGUF reale.
- Evidenza positiva sufficiente come accettazione personale ai sensi
  della nuova MIG-FLOW-002 revisione 2; l'utente ha autorizzato
  espressamente la chiusura automatica senza una seconda conferma.
- Nessun GGUF reale, dato tensore, dequantizzazione, CUDA,
  backend, download o inferenza collaudati. Nessun dato host,
  path, output grezzo o log pubblicato (DATA-001).
- GGUF-KBLOCK-001: implementato, CI verificata, integrato;
  manual accepted. Le precedenti note pending sono storiche.
  MIG-FLOW-002-R2 elimina il secondo via libera dopo prova positiva,
  ma MIG-FLOW-001 richiede approvazione dello scope futuro.

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

## MIG-013 / GGUF-STRUCT-SUMMARY-SCOPE-001 approvata (17 ottobre 2026)

- Approvazione esplicita dell'utente; base main `2a2ea80eee7b35c9c2d195e6c0de66ac57ee6550`.
- Contratto read-only: riepilogo per soli GGUF dichiarati in CAT dei lettori
  GGUF, META, INDEX, BLOCKS e KBLOCKS; stati `structural_checked`,
  `size_unverified`, `invalid`, `not_configured` (skip valido).
  L'esito INDEX/BLOCKS `size_unverified` puo' essere superato solo dalla
  verifica completa degli ingombri operata da KBLOCKS; gli sconosciuti
  rimangono `size_unverified`, errori strutturali diventano `invalid`.
- Tredicesimo passaggio del default Ruth `SUMMARY`, dopo FIXTURE;
  `gguf-summary` per ispezione mirata. Fixture sintetiche e test di regressione,
  CI e promozioni GGUF -> Formats -> Models -> main richiesti.
- Stato **implementazione candidata**, verifica **pending**, integrazione
  **pending**, `GGUF-STRUCT-SUMMARY-MANUAL-001` **pending** dopo merge.
  Collegamento locale: `local_sync: pending_no_access`.
- Nessuna lettura di pesi, download, rete, checksum payload, inferenza, CUDA
  o benchmark. DATA-001, REMOTE-001, MIG-FLOW-001/002 invariati.
  Collaudo personale positivo sufficiente senza seconda conferma.

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

## MIG-014 / GGUF-STRUCT-STABILITY-001 approvata (17 ottobre 2026)

- Utente ha approvato il perimetro proposto; commit base main
  `5a0a912d47e5860c8dcb8fd0188b6b24913314b1`.
- Estensione limitata alla fase SUMMARY già presente: osservazione dei
  metadati del file GGUF dichiarato prima/dopo lettori bounded.
  Identità (dispositivo/inode), dimensione, mode/link e timestamps ns;
  se cambiano: `invalid` / `STABILITY:changed_during_inspection`;
  se non verificabili o divenuti indisponibili: `invalid` /
  `STABILITY:unavailable`. I report privi di differenze osservate
  conservano lo stato originale, anche `size_unverified`.
- Solo test sintetici per file stabile, sostituzioni, variazioni metadati,
  rimozione e symlink. 13 fasi Ruth invariate; nessun payload letto.
- Non è una garanzia atomica di immutabilità o hash/integrità crittografica.
  Non copre modifiche non rilevate dai due snapshot. DATA-001 e
  REMOTE-001 invariati, Kaggle inattivo.
- Stato attuale: scope **approved**, implementazione **candidate**,
  verifica **pending**, integrazione **pending**, collaudo personale
  `GGUF-STRUCT-STABILITY-MANUAL-001` **pending** dopo merge.
  Branch `models/formats/gguf/work/GGUF-STRUCT-STABILITY-001`.
  Local checkout non accessibile: `local_sync: pending_no_access`.

## MIG-014 integrazione tecnica completata (17 ottobre 2026)

- PR #84 (GGUF), #85 (Formats), #86 (Models), #87 (main) integrate con
  GitHub CI **success** su ogni promozione (test sintetici, regressioni,
  punto d'ingresso e mappe). Commit applicativo `1a77914e1e3ed9ee7cdbe7a57c2ddccffce7b616`.
- Il controllo file stabile è in SUMMARY, **13 fasi Ruth invariate**.
  Confronta attributi e identità osservabili prima e dopo la sequenza di
  lettori già autorizzati; file cambiato -> `invalid` /
  `STABILITY:changed_during_inspection`, file divenuto indisponibile ->
  `invalid` / `STABILITY:unavailable`. Un file invariato mantiene i
  risultati precedenti anche `size_unverified`.
- Suite `test_stability.py`: **9 test sintetici** passati in CI;
  simulati mutazione dimensionale, sostituzione stesso ingombro, cambi
  timestamp, cancellazione, symlink, casi stabili validi e invalidi.
- Stato: implementation **complete**, verification **passed (CI)**,
  integration **merged**; **GGUF-STRUCT-STABILITY-MANUAL-001 pending**.
  Prova personale post-merge: default Ruth 13 fasi e 9 test mirati `test_stability`
  su commit main verificato. L'evidenza positiva equivale ad accettazione
  senza doppia conferma (MIG-FLOW-002 rev.2).
- Nessun peso o modello reale verificato, nessun hash payload; il confronto
  non garantisce immutabilità atomica né integrità crittografica. Nessuna
  lettura tensor_data, rete, download, CUDA, inferenza o benchmark.
  DATA-001 e REMOTE-001 invariati; `local_sync: pending_no_access`.
  Nessuna nuova migrazione approvata.

## GGUF-STRUCT-STABILITY-MANUAL-001 accettata; MIG-014 chiusa (17 ottobre 2026)

- Prova personale ricevuta sull'esatta revisione main `b1662aab99cbc97b6203dc41e7355521e702a2cd`,
  verificata identica a GitHub heads/main prima della chiusura.
- `main.py` senza argomenti: **13 fasi completate** in ordine senza errori;
  fixture GGUF sintetiche: **passed**, tre file. SUMMARY:
  `not_configured` / `CAT:not_configured` con catalogo opzionale assente,
  skip valido; non e' stata effettuata ispezione di un GGUF reale.
- `models.formats.gguf.tests.test_stability`: **9 test, OK**,
  zero fallimenti, sui controlli sintetici di mutazione/stabilita'.
- La prova positiva e' accettazione sufficiente in MIG-FLOW-002 rev.2,
  **senza ulteriore consenso**. MIG-014: implementation **complete**,
  verification **passed** (CI + collaudo personale), integration
  **merged** (PR #84-#87), personal acceptance **passed**.
- Non sono stati certificati pesi, hash, immutabilita' atomica, integrita',
  modelli reali, CUDA, backend o inferenza: snapshot best-effort soltanto.
- DATA-001: registrati solo SHA, stati e conteggi; nessun log grezzo,
  inventario host, percorso personale, credenziale o benchmark in Git.
- Revisione locale riferita dal collaudo utente: `local_sync: verified_on_tested_revision`.
  Dopo il merge documentale e' necessario il fast-forward dei soli documenti:
  `local_sync: pending_docs_fast_forward` sino a nuova prova; non si presume
  verificato uno stato Git pulito del checkout personale.
- MIG-015 non approvata: presentare perimetro funzionale distinto per
  autorizzazione ai sensi di MIG-FLOW-001. Le precedenti note `pending`
  sono cronologia storica, non stato corrente.

## MIG-015 / GGUF-FIXTURE-SUMMARY-001 approvata (17 ottobre 2026)

- Utente ha approvato esplicitamente la porzione MIG-015 dopo il collaudo MIG-014.
  Commit base main `0cb12ad90e2ef7d1b3b3b77009f9cf476a3e5289`.
- La fase FIXTURE legge anche SUMMARY sul catalogo *solo temporaneo* dei tre
  GGUF sintetici. Richiesti: `SUMMARY=structural_checked`, tre modelli con
  `structural_checked`, conteggi e ragioni esatti; esito anomalo ->
  `failed` con motivazione portabile. Gli stati INDEX/BLOCKS parziali legacy
  restano legittimi quando KBLOCKS verifica i formati noti.
- Nuova suite mirata `test_fixture_summary` (sette test sintetici) con
  formati sconosciuti, magic invalida, truncation, mutazioni e fallimenti
  indotti di SUMMARY. Regressione `test_pipeline` allineata.
- Ruth conserva **13 fasi**, la fase SUMMARY finale sul catalogo reale resta
  separata dalla SUMMARY temporanea interna a FIXTURE.
- Stato iniziale: scope **approved**, implementazione **candidate**,
  verification **pending**, integration **pending**, manual gate
  `GGUF-FIXTURE-SUMMARY-MANUAL-001` **pending** post-merge.
- Nessun modello reale, payload, download, rete, benchmark, CUDA o inferenza;
  DATA-001, REMOTE-001, MIG-FLOW-001/002 vigenti; Kaggle inattivo.
