# Formats: checkpoint GGUF-001

- Contratto formato minimo delegato al lettore GGUF header-only.
- Stato candidato: codice foglia in sviluppo/verifica, CI e promozioni pendenti.
- Nessun parser completo, GGUF verification, acquisizione o inferenza autorizzati.
- GGUF-MANUAL-001 richiesto solo dopo integrazione in main.

## GGUF-001 integrata e verificata (16 ottobre 2026)

- API di lettura intestazione GGUF e catalogo delegato verificate in CI,
  promosse via PR #41-#44 sino a main `15b106082c4dee37622c61a8afefd248a57464a0`.
- Test sintetici e mappe coerenti superati; nessuna ispezione di pesi o
  metadati completi. Il dominio Formats resta solo parzialmente attivo.
- GGUF-MANUAL-001 e' pending; prossimo componente solo dopo la prova
  dell'utente e l'approvazione del nuovo scope.

## GGUF-MANUAL-001 accettata (16 ottobre 2026)

- L'utente ha confermato positivamente l'avvio personale di Ruth
  in sette fasi sul commit main `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`.
- Catalogo assente e controllo GGUF non configurato: skip corretto.
  Nessun file GGUF reale ispezionato nel collaudo.
- GGUF-001 implementata, CI verificata, integrata, accettata manualmente.
  Le note di attesa precedenti sono storiche.
- Nessun nuovo perimetro autorizzato; DATA-001 resta valido.

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
