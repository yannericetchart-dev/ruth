# GGUF-STRUCT-SUMMARY-SCOPE-001 — MIG-013

Approvazione: esplicita dell'utente, 17 ottobre 2026. Base main: `2a2ea80eee7b35c9c2d195e6c0de66ac57ee6550`.

## Funzioni autorizzate

- `inspect_structural_summaries(*, project_root: Path | None = None) -> GGUFStructuralReport`: riepiloga solo i GGUF elencati dal catalogo CAT, senza cercare file né modificare dati.
- Per ciascun modello GGUF usare i lettori bounded già integrati GGUF, META, INDEX, BLOCKS e KBLOCKS. Includere header, dichiarazioni tecniche, descrittori, ingombro byte dei tipi noti, offset, allineamento, intervalli/sovrapposizioni già controllati dai lettori esistenti.
- Esiti per modello e aggregati: `structural_checked`, `size_unverified`, `invalid`; `not_configured` quando non ci sono GGUF da esaminare (catalogo assente, vuoto oppure nessun GGUF). Motivi portabili, niente nomi di tensori o riferimenti al filesystem nei report.
- Validità: INDEX/BLOCKS restano `size_unverified` per tipi quantizzati sconosciuti a quei lettori; l'aggregatore può dichiarare `structural_checked` **solo** quando KBLOCKS verifica gli ingombri di ogni tensore supportato e tutte le verifiche preliminari sono coerenti. Se KBLOCKS non può verificare un tipo, l'esito è `size_unverified`, mai falso successo. Errore di un lettore ha precedenza e produce `invalid`.
- Tredicesimo passaggio Ruth `SUMMARY` dopo FIXTURE e comando mirato `gguf-summary`. Nessuna modifica agli esiti storici e alle 12 fasi esistenti. Catalogo assente = skip non bloccante; `invalid`/`size_unverified` = esito nonzero.
- Test con fixture sintetiche di scalare, Q4_0, Q4_K, tipo sconosciuto, header malformato, estensione troncata, catalogo mancante/non valido, report immutabili e CLI. CI e mappe ricorsive, PR lungo GGUF -> Formats -> Models -> main, collaudo personale post-merge.

## Esclusioni e garanzie

Nessuna lettura dei byte dei pesi, dequantizzazione, checksum del payload, download, rete, CUDA, backend, inferenza, installazioni o benchmark. Niente nuovi modelli reali nel repository. I codici di stato sono dichiarazioni **solo strutturali**, non attestati di integrità o compatibilità. DATA-001, REMOTE-001 e MIG-FLOW-001/002 vigenti. Kaggle inattivo.

Stato all'avvio: approved; implementation candidate; CI/integration/manual pending.
