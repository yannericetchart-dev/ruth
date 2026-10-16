# CAT-SCOPE-001 — Catalogo locale dichiarativo (approvato)

L'utente ha approvato il perimetro CAT-SCOPE-001 il 16 ottobre 2026,
sulla revisione iniziale `60e137a422403efa7d8d4005240abf317ce87e24`.
Fase 3, migrazione per componente, MIG-FLOW-001/002.

## Interfaccia e formato

- `models.catalog.catalog(*, project_root=None) -> ModelCatalog`: legge
  solo il file locale opzionale `.local/models/catalog.json`, sotto
  la radice del repository. Se manca, ritorna `not_configured`;
  se dichiara zero modelli, `empty`. Nessuna scansione ricorsiva.
- JSON previsto: `{"models":[{"id":"demo","format":"gguf","path":"demo.gguf"}]}`.
  Il valore `path` e' sempre relativo a `.local/models/`, mai
  assoluto, con traversal, segmenti nascosti o link simbolici.
  L'utente crea il manifest e dispone personalmente dei modelli: Ruth
  non crea directory, non scrive file e non scarica niente.
- Ogni `ModelRecord(id, format, reference, status)` e' immutabile;
  stato `present_unverified` oppure `missing`. Il campo `format`
  e' solo dichiarato: NON significa GGUF verificato. Il riferimento
  e' relativo alla cartella locale, non un path assoluto host.
- `ModelCatalog(status, models, errors)` e' immutabile e transitorio.
  Stati: `not_configured`, `empty`, `ready`, `partial`,
  `invalid`. Gli errori sono codici portabili, non testo privato
  o contenuto del manifest. Dimensione massima del JSON 1 MiB e
  al massimo 200 voci. Formato identificatori e percorsi limitato.
- `models.catalog.resolve(reference, *, project_root=None)` restituisce
  un `ModelRecord` per id esplicito oppure `None` se non dichiarato;
  alza `ValueError` per catalogo o riferimento non valido.
- `main.py` senza argomenti: PY -> DEP -> DIAG -> HW CPU/RAM ->
  NVIDIA -> CAT, anche se le condizioni Python/DEP sono errate.
  `not_configured` e `empty` non sono errori; manifest non valido o
  file mancante restituisce uno stato diagnostico nonzero senza
  impedire la visualizzazione dei controlli precedenti.
  `main.py catalog` esegue solo il controllo mirato del catalogo.

## Esclusioni e verifica

- Non leggere contenuti GGUF, tensor, metadati, licenze, dimensioni,
  né caricare modelli in memoria; nessuna ricerca filesystem,
  chiamata di rete, Hugging Face, installazione o inferenza.
- DATA-001: `.local/` resta ignorata da Git; mai versionare manifest
  reale, elenco dei modelli posseduti, percorsi, log o dati macchina.
- Test su fixture temporanee sintetiche: no config, vuoto, file
  presente/mancante, duplicati, JSON invalido, assoluti/traversal,
  symlink di file/cartelle/manifest, dimensioni e ordine CLI.
  Mappe e CI su GitHub prima di ogni promozione.
- Promozione approvata: `models/catalog/work/CAT-001` ->
  `models/catalog/main` -> `models/main` -> `main`.
  Dopo main richiedere collaudo personale CAT-MANUAL-001.

**Stato iniziale:** codice candidato; CI e integrazione pending.
