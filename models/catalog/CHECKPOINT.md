# Catalogo locale: checkpoint CAT-001

## Stato del ref remoto (17 ottobre 2026)

Branch logico `models/catalog/main`: **archived / ref remoto assente**.
Tip prima della rimozione: `90e57f106fb664b0257775260044125ea85228e2`.
Rimosso dall'esecutore in BRANCH-DELETE-002 (PR #109), verificato da B
tramite inventari completi heads/branches main-only. Conservazione: storia contenuta in main alla revisione
`e7efb79a86e28209528ebcf4d85ca41f526b9b39`.
NODE aggiornato ad archived, flag archived_branch e viste rigenerate.
Nomi e padri Git sotto restano provenienza e contratti logici, non ref presenti.
Codice, API e collaudi applicativi conservano il loro ambito precedente.
Per la coda documentale e lo stato finale leggere CHECKPOINT.md radice e
il checkpoint BRANCH-CLEANUP-001-FINAL sulla PR della registrazione.


**CAT-SCOPE-001 implementato, verificato e integrato in main**.
Approvazione 16 ottobre 2026. CAT-MANUAL-001 superato personalmente.

- Contratti `ModelCatalog` e `ModelRecord` immutabili, funzioni
  `models.catalog.catalog` e `models.catalog.resolve`.
- File opzionale `.local/models/catalog.json`, dati locali esclusi
  da Git. Stati not_configured / empty / ready / partial / invalid.
  Manifest vuoto o non configurato non interrompe il default Ruth.
- Modelli elencati con id, formato dichiarato, riferimento relativo e
  stato del file; nessuna lettura contenuto del modello, scansione
  generale, download, import modello o verifica GGUF.
- Default Ruth PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT; `main.py
  catalog` per controllo mirato, senza operazioni precedenti.
- PR #36 -> models/catalog/main
  (`a562e1b4ce948d3865a15a9d06454d91b8cb7cde`),
  PR #37 -> models/main
  (`77e3a7327c9f0750d6e8363766000ba830ea1c5e`),
  PR #38 -> main
  (`12c395ba3bb272d66095d4e468794936b1693892`).
- CI GitHub 38928909391, 38929028829, 38929118836: success,
  116 fixture sintetiche e 264 mappe coerenti.
- **CAT-MANUAL-001 complete**: conferma personale positiva sul commit
  `90e57f106fb664b0257775260044125ea85228e2`, con sesta fase CAT e stato `not_configured` valido.
  Nessuna migrazione successiva senza nuovo perimetro approvato.
- REMOTE-001 e DATA-001: vecchio checkout intatto; nessun manifest,
  percorso, inventario o modello reale in Git.

## CAT-MANUAL-001: conferma positiva (16 ottobre 2026)

- L'utente ha sincronizzato la copia locale pulita fino al commit main
  `90e57f106fb664b0257775260044125ea85228e2`, ha eseguito `.venv/bin/python main.py`
  senza argomenti e ha confermato esplicitamente il collaudo positivo.
- Sesta fase catalogo `not_configured` non bloccante, insieme ai cinque
  controlli precedenti e al riepilogo conclusivo regolare.
- Stato CAT-001: complete/passed/merged; accettazione manuale passed.
  Le note precedenti con `active` o `pending` sono storiche.
- Non sono state aggiunte capacita' GGUF, download o inferenza;
  nessun dato reale e' versionato. Il prossimo scope e' da approvare.
