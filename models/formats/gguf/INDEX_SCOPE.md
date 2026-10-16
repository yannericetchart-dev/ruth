# GGUF-INDEX-SCOPE-001 — struttura dell'indice tensori

Approvazione esplicita dell'utente: 16 ottobre 2026 dopo
GGUF-META-MANUAL-001 positivo, main base `240e1bc8616f9087390c08603704db4165ea1587`.
MIG-009; perimetro read-only della nona fase Ruth.

## Contratti

- Ingressi: soltanto gli ID dei GGUF elencati e validati dal catalogo
  CAT-001 opzionale in `.local/models/catalog.json`. Nessuna scansione,
  percorso arbitrario, rete o acquisizione.
- `inspect_tensor_index(model_id, *, project_root=None) -> GGUFTensorIndex`
  e `inspect_all_tensor_indexes(*, project_root=None) -> GGUFTensorIndexReport`.
  Report immutabili con ID, stato, conteggio descrittori e numero di
  tipologie il cui ingombro non e' certificabile; nessun nome reale
  o contenuto di modello nel risultato.
- Parser v2 little-endian, v3 little-/big-endian: intestazione GGUF,
  metadati KV bounded, `general.alignment` o valore implicito 32,
  descrittori contenenti nome, `n_dimensions`, dimensioni, codice tipo
  e offset relativo all'origine dei dati tensore. La sezione dati
  non viene letta e il padding non viene esaminato.
- Verifiche: nomi UTF-8 di massimo 64 byte e non duplicati;
  numero e prodotto delle dimensioni positivo e limitato (massimo
  4 dimensioni); offset unici e allineati, relativi alla sezione
  dati e inferiori alla dimensione del file; per i soli tipi scalari
  con dimensione elementare nota, intervalli entro il file e senza
  sovrapposizioni con offset successivi. Non controlla contenuti.
- Codici portabili: `index_checked` soltanto quando i controlli
  supportati passano; `size_unverified` se almeno un tipo non
  ha ingombro certificabile (incluse quantizzazioni), con esito
  aggregato `partial`, mai dichiarato completamente validato.
  Errori specifici per tipo/versione, magic, truncation, duplicati,
  dimensioni, allineamento, offset, overlap, catalogo e percorsi.
- Limiti: massimo 16.384 tensori e 16.384 metadati, 64 MiB di
  strutture esaminate complessivamente, massimo 64 byte per nome,
  prodotto dimensioni fino a 2^63-1; allineamento massimo 1 MiB.
  Metadati e array soggetti ai limiti bounded di GGUF-META-001.
- Effetti: sola lettura con controlli dei symlink e O_NOFOLLOW,
  nessun file modificato. Nessun peso letto, nessun tensore in
  memoria, nessun hash, verifica completa del formato, compatibilita'
  per inferenza o installazione.
- CLI: nono passaggio di `main.py` dopo GGUF-META, piu' sottocomando
  mirato `main.py gguf-index`. Il catalogo assente, vuoto o senza
  GGUF e' uno skip valido e non bloccante.
- Test: fixture sintetiche di dimensioni/allineamento/offset,
  file troncati, nomi duplicati, tipi sconosciuti, versioni/endianness,
  percorsi non sicuri, metadati bounded e sequenza Ruth.
- Verifica: CI, mappe ricorsive, promozioni gguf -> formats -> models ->
  main; poi GGUF-INDEX-MANUAL-001 personale e conferma esplicita.

Vincoli: MAIN-FLOW-001, MIG-FLOW-001/002, REMOTE-001, DATA-001.
Esclusi: dati reali Git, download, Hugging Face, CUDA,
benchmark, llama.cpp e inferenza. Kaggle inattivo.
Stato iniziale: approvato, candidato non ancora verificato/integrato.
