# GGUF-META-SCOPE-001 — metadati GGUF dichiarati (approvato)

Approvazione funzionale esplicita dell'utente il 16 ottobre 2026, base
main \`044f0504b8e023689c0b78bc981440fbc06aee12\`.
Il perimetro amplia GGUF-001 (solo intestazione) senza modificare il
contratto di header e senza includere payload dei tensori.

## Funzioni

- Ingressi: un id di modello dichiarato in CAT-001 oppure l'intero
  catalogo validato; nessun percorso libero, ricerca o download.
- \`inspect_metadata(model_id, *, project_root=None) -> GGUFMetadata\`
  restituisce stato e tre informazioni opzionali:
  \`general.architecture\` (stringa), \`general.file_type\` (intero
  dichiarato) e \`{architecture}.context_length\` (intero dichiarato).
  Nessuna deduzione su quantizzazione/compatibilita': il codice file_type
  non viene convertito in una classe di quantizzazione.
- \`inspect_all_metadata(*, project_root=None) -> GGUFMetadataReport\`
  restituisce tuple immutabili. Stati catalogo assente/vuoto/senza GGUF
  sono validi, senza fallimento. Altri errori hanno codici portabili
  senza testo binario, path utente o valori privati.
- File GGUF v2 little-endian e v3 little-/big-endian. Scanner delle
  coppie KV con tipi scalari/stringhe/array (annidati) solo per saltare
  i valori estranei e individuare le tre informazioni approvate.
- Limiti: massimo 64 MiB di descrittori letti/saltati per file,
  16.384 coppie, 8 MiB per stringa, 500.000 elementi in ciascun
  array, un milione complessivi e 6 livelli di array. Nessuna
  lettura contenuto di tensori. File enormi o malformati possono
  restituire \`limit_exceeded\` (non crash, non falso successo).
- Sicurezza: riferimenti e symlink controllati dal catalogo e di
  nuovo prima dell'apertura; sola lettura con O_NOFOLLOW quando presente,
  controllo lunghezza rispetto a fstat, no rete/scritture/installazioni.
- Ingresso ufficiale: ottava fase di \`main.py\` senza argomenti,
  dopo GGUF; sottocomando \`main.py gguf-metadata\` per prova mirata.
  Output e codici chiari anche su errori e catalogo assente.
- Esclusi: verifica intera struttura GGUF, peso tensor, mmap dei pesi,
  llama.cpp, download, Hugging Face, CUDA, runtime/inferenza e benchmark.
  Kaggle inattivo; DATA-001 applicato a tutti i dati reali.

## Verifica e consegna

Test sintetici GGUF v2/v3, endian, stringhe, array annidati, troncamenti,
tag invalidi, key duplicates, overflow e limiti, symlink, catalogo
non configurato, ordine e sottocomando CLI. CI GitHub, mappe generate
da NODE e promozioni attraverso gguf -> formats -> models -> main.
Dopo l'integrazione resta GGUF-META-MANUAL-001 personale: esecuzione
di \`.venv/bin/python main.py\` da main sincronizzato e conferma esplicita
prima di proporre una migrazione ulteriore.

Stato iniziale: approvato; codice candidato; CI e integrazione non ancora
verificate; il perimetro non e' una certificazione di compatibilita'.
