# GGUF-BLOCK-SCOPE-001 — ingombri di cinque tipi GGML

Approvazione esplicita dell'utente: 17 ottobre 2026, main base
`6f472b0ac182b113c5c727c3f8f5c291c074de2e`. MIG-010; perimetro limitato al controllo
read-only della struttura degli indici tensore dei GGUF dichiarati da CAT.

## Contratto approvato

- `inspect_block_layout(model_id, *, project_root=None) -> GGUFBlockLayout`
  e `inspect_all_block_layouts(*, project_root=None) -> GGUFBlockLayoutReport`;
  risultati immutabili contenenti solo ID dichiarato, codici e conteggi.
- Riuso del parser bounded di GGUF-INDEX-001 (v2/v3, endian, KV,
  allineamento, nomi, dimensioni e offset relativi). Nessun percorso
  esterno al catalogo, symlink o directory traversal consentiti.
- Cinque GGML type ID certificati con definizioni note:
  Q4_0=2 (32 elementi, 18 byte/blocco),
  Q4_1=3 (32, 20),
  Q5_0=6 (32, 22),
  Q5_1=7 (32, 24),
  Q8_0=8 (32, 34). Fonte: include/ggml.h, src/ggml-common.h
  del progetto upstream ggml. Per questi soli tipi, la dimensione
  interna (prima dimensione) deve essere divisibile per 32;
  ingombro = prodotto delle dimensioni / 32 * byte per blocco.
- Tipi scalari gia' supportati dalla fase INDEX restano verificati
  con la loro ampiezza nota; per altri tipi o quantizzazioni
  l'ingombro resta `size_unverified` e il report aggregato `partial`.
  Non inventare supporto e non restituire success completo per casi
  parzialmente verificabili.
- Controlli: divisibilita', dimensioni positive e prodotto limitato,
  offset multipli dell'allineamento, intervalli entro il file,
  non sovrapposizione, tipi non conosciuti, riferimenti insicuri,
  catalogo invalido e file troncati. Risultato `layout_checked`
  soltanto quando tutti i descrittori hanno dimensione attestabile
  e controlli strutturali superati.
- Limiti ereditati dalla fase INDEX: massimo 16.384 descrittori e
  metadati, 64 MiB di struttura letta/saltata, 64 byte per nome,
  massimo 4 dimensioni e prodotto fino a 2^63-1;
  nessun accesso ai bytes dei tensor_data.
- CLI `main.py` avvia tutti e dieci i passaggi default:
  PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF -> META ->
  INDEX -> BLOCKS; sottocomando mirato `gguf-blocks`.
  Catalogo assente/vuoto/senza GGUF: skip valido non bloccante.
- Verifica con fixture sintetiche di cinque tipi e diverse
  endianness, multipli di blocco, offset, bounds, overlap,
  malformed, no-git-private-data; regressioni CLI e lettore INDEX.
- Esclusi payload, lettura o dequantizzazione pesi, hash,
  verifica integrita' completa, inferenza, backend, CUDA,
  Hugging Face, rete, acquisizione, download, installazione e
  benchmark. Kaggle inattivo; DATA-001, REMOTE-001 invariati.
- Promozioni via gguf -> formats -> models -> main, CI e mappe
  coerenti. GGUF-BLOCK-MANUAL-001 richiede conferma positiva personale
  post-merge prima di una nuova migrazione (MIG-FLOW-002).

Stato iniziale: scope approvato, implementazione candidata,
CI e integrazione non ancora verificate.