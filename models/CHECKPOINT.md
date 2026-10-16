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
