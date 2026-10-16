# Catalogo locale: checkpoint CAT-001

**CAT-SCOPE-001 implementato, verificato e integrato in main**.
Approvazione 16 ottobre 2026. Manual acceptance ancora in attesa.

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
- **CAT-MANUAL-001 active**: dopo fast-forward non distruttivo della
  copia pulita, confermare revisione e sesto messaggio della CLI.
  Nessuna migrazione successiva prima dell'accettazione.
- REMOTE-001 e DATA-001: vecchio checkout intatto; nessun manifest,
  percorso, inventario o modello reale in Git.
