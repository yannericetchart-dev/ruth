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
