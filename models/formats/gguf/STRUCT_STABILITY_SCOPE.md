# GGUF-STRUCT-STABILITY-SCOPE-001 — MIG-014

Approvata esplicitamente dall'utente il 17 ottobre 2026, dopo MIG-013 completa. Base main `5a0a912d47e5860c8dcb8fd0188b6b24913314b1`.

## Perimetro funzionale

- Rafforzare **soltanto** la fase SUMMARY già integrata, senza creare una quattordicesima fase.
- Su ciascun GGUF dichiarato e presente nel catalogo CAT, registrare prima della sequenza dei lettori bounded e dopo il loro completamento una fotografia di metadati filesystem: identificativi dispositivo/inode, modalità e numero link, dimensione, mtime_ns, ctime_ns.
- Riconvalidare il percorso secondo i controlli symlink e riferimenti già esistenti; snapshot assente, file non regolare o percorso diventato indisponibile: esito per-modello `invalid`, codice portabile `STABILITY:unavailable`.
- Snapshot diversi: `invalid`, motivo `STABILITY:changed_during_inspection`, con precedenza su un apparente `structural_checked` e sugli stati della verifica del file già eseguita. Nessuna modifica degli stati degli altri modelli aggregati.
- Snapshot uguali: conservare il precedente esito `structural_checked`, `size_unverified` oppure `invalid` con motivo originario. Catalogo non configurato rimane skip valido.
- Prove solo temporanee/sintetiche: file invariati, aggiunta di byte durante ispezione, sostituzione con inode differente anche a parità di dimensione, cambiamento metadati senza modificare dimensione, rimozione o symlink durante la sequenza, tipi sconosciuti e file invalidi stabili.
- CI, mappe ricorsive, promozioni GitHub GGUF -> Formats -> Models -> main; poi collaudo personale con `main.py` (13 fasi) e `test_stability.py` (9 test), accettazione automatica su prova positiva secondo MIG-FLOW-002 revisione 2.

## Limiti

Questo è un **rilevatore best-effort di modifiche osservabili**, non un blocco/snapshot atomico del file e non una certificazione crittografica: modifiche ripristinate tra le rilevazioni, file-system con timestamp poco precisi o gare non osservabili possono sfuggire. Nessuna lettura pesi, hash di payload, modello reale, inferenza, CUDA, download o rete. Nessun output host/utente registrato in Git (DATA-001); REMOTE-001 vigente. Kaggle inattivo.

Stato: scope **approved**; implementation candidate; CI/integration/manual pending.
