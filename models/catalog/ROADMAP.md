# Catalogo / models/catalog

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CAT](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Unificare descrizioni e riferimenti dei modelli.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: not_run; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: CAT-001: manifest JSON opzionale in .local/models/catalog.json, file dichiarati sotto .local/models senza symlink o traversal, report read-only, niente discovery/download/GGUF o dati reali su Git. Test sintetici e promozioni ancora da eseguire.

## Attivita locali

1. `CAT-001` [active] Manifest locale JSON rigoroso, report immutabili ModelCatalog/ModelRecord, protezione directory/symlink e controllo file dichiarati; CI e promozione pending.
2. `CAT-SCOPE-001` [complete] Catalogo locale dichiarativo read-only, percorso manifest opzionale, casi invalidi e sesto passaggio approvati esplicitamente il 16 ottobre 2026.
3. `CAT-MANUAL-001` [planned] Dopo merge su main, test personale dal flusso main.py completo con catalogo not_configured/empty oppure manifest locale; comunicare SHA e stato, nessun riferimento host in Git.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
