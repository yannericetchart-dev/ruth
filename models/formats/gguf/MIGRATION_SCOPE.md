# GGUF-SCOPE-001 — lettura limitata di intestazioni GGUF

- Approvazione funzionale esplicita dell'utente: 16 ottobre 2026, dopo
  accettazione CAT-MANUAL-001 e merge PR #40, base `022928e56c5915dc4a9cb0850d62de24972ec190`.
- Perimetro: leggere esclusivamente i primi 24 byte dei file con `format:
  "gguf"` presenti nel catalogo opzionale e gia' validati da CAT-001.
  Nessun path libero utente, scansione, download, rete, installazione,
  lettura metadata payload o tensori, caricamento o inferenza.
- Contratto: `inspect_header(model_id, *, project_root=None)` restituisce
  `GGUFHeader` immutabile e codici di errore portabili; `inspect_headers`
  produce `GGUFReport` per tutte le dichiarazioni GGUF esistenti. Un
  catalogo assente/vuoto o senza GGUF e' un esito valido di skip.
- Campi letti: magic `GGUF` (4 byte), versione uint32, contatori uint64
  per tensor e metadati. Versioni 2 e 3; lettura little-endian e big-endian
  v3. Riferimento: specifica ggml GGUF, https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- Stati diagnostici: `header_read`, `missing`, `unreadable`,
  `invalid_magic`, `truncated_header`, `unsupported_version`,
  `unsafe_reference`, `not_declared` e `invalid_catalog`.
  La sola intestazione corretta NON certifica integrita' del modello,
  completezza del file, compatibilita' o runtime.
- Ingresso ufficiale: settima fase in `main.py` senza argomenti,
  PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF, con
  riepilogo e codici d'uscita coerenti; comando mirato `main.py gguf`.
  Le fasi indipendenti restano osservabili anche quando Python fallisce.
- Test: fixture sintetiche per v2/v3 e endian, troncamenti, firme,
  versioni, percorsi invalidi, symlink, catalogo non configurato/vuoto
  e ordine nella CLI; validare `python3 tools/maps.py --check` e GitHub CI.
- Sviluppo GitHub-first su branch foglia, promozioni verso
  `models/formats/gguf/main` -> `models/formats/main` ->
  `models/main` -> `main`. Dopo integrazione e sync sicura,
  `GGUF-MANUAL-001` richiede prova locale dell'utente e conferma positiva.
- DATA-001: nessun nome reale di file/modello, path, log, inventario o
  contenuto binario in Git; Kaggle rimane inattivo.

Stato iniziale: candidate implemented; verification not_run; integration
not_merged. Non confondere tale stato con disponibilita' gia' in main.
