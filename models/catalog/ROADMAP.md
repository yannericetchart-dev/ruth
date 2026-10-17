# Catalogo / models/catalog

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CAT](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Unificare descrizioni e riferimenti dei modelli.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope, archived_branch. Blocchi: nessuno registrato.

Ambito verifica: CAT-001: CI GitHub 38928909391, 38929028829, 38929118836 success (116 test sintetici inclusi 21 catalogo e 20 CLI, 264 mappe). PR #36 models/catalog/main, #37 models/main e #38 main. Manifest locale read-only, file solo dichiarati, niente download, lettura GGUF, inferenza o dati reali Git. CAT-MANUAL-001 confermato positivo sul commit 90e57f106fb664b0257775260044125ea85228e2.

## Attivita locali

1. `CAT-001` [complete] Catalogo locale dichiarativo con validazione/sicurezza percorsi verificato da CI e integrato via PR #36-#38 in main.
2. `CAT-SCOPE-001` [complete] Catalogo locale dichiarativo read-only, percorso manifest opzionale, casi invalidi e sesto passaggio approvati esplicitamente il 16 ottobre 2026.
3. `CAT-MANUAL-001` [complete] Collaudo personale positivo di main.py senza argomenti sul commit 90e57f106fb664b0257775260044125ea85228e2: sesta fase catalogo con not_configured valido; confermato il 16 ottobre 2026.
4. `CAT-PROMOTE-001` [complete] PR #36 integrata in models/catalog/main; merge a562e1b4ce948d3865a15a9d06454d91b8cb7cde.
5. `CAT-PROMOTE-002` [complete] PR #37 integrata in models/main; merge 77e3a7327c9f0750d6e8363766000ba830ea1c5e.
6. `CAT-PROMOTE-003` [complete] PR #38 integrata in main; merge 12c395ba3bb272d66095d4e468794936b1693892.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
