# GGUF-001: checkpoint di candidato

- GGUF-SCOPE-001 approvato il 16 ottobre 2026 su main `022928e56c5915dc4a9cb0850d62de24972ec190`.
- Implementazione lettore e settima fase candidate sul work branch;
  verifiche CI e promozioni completate; collaudo personale ancora da eseguire.
- GGUF-MANUAL-001 planned post-merge; catalogo assente -> skip valido.
- Nessuna operazione GGUF oltre la sola lettura dei 24 byte iniziali.
- DATA-001 e REMOTE-001 invariati. Non pubblicare dati locali.

## GGUF-001 integrata (16 ottobre 2026)

- PR #41, #42, #43, #44 integrate sino al commit main `15b106082c4dee37622c61a8afefd248a57464a0`.
- CI runs 38935705385, 38935861881, 38935943414, 38936032188
  **success**: test intestazioni v2/v3 endian, errori e percorsi, ingresso
  Ruth con settimo passaggio automatico, mappe coerenti.
- Stato finale tecnico: implementation complete, verification passed
  (solo fixture sintetiche), integration merged.
- **GGUF-MANUAL-001 active/pending**: richiede conferma esplicita
  dell'utente dopo esecuzione locale da main aggiornato, anche quando
  il catalogo non e' configurato e la settima fase e' correttamente saltata.
- Il solo header non certifica l'integrita' di un modello GGUF o
  l'idoneita' all'inferenza. Nessun dato privato o binario in Git.

## Chiusura GGUF-MANUAL-001 (16 ottobre 2026)

- Conferma esplicita positiva dell'utente dopo avvio locale ufficiale
  con `.venv/bin/python main.py` sul commit
  `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`.
- Settima fase GGUF `not_configured` nel flusso di sette controlli:
  stato previsto con catalogo assente. Non e' stata verificata
  la lettura di un file GGUF reale.
- GGUF-001: implementation complete, verification passed (test sintetici),
  integration merged, **manual acceptance passed**.
- GGUF-MANUAL-001 complete; le note pending precedenti sono storiche.
  Prossimo componente solo dopo nuovo scope esplicitamente approvato;
  nessun dato macchina o modello incluso in Git (DATA-001).

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
