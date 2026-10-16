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
