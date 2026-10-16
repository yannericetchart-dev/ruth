# Contratti e orchestrazione / runtime

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CORE](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Esporre i contratti comuni e coordinare i domini.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: PY-001: GitHub Actions run 38885972247 e run 38886317198 conclusi con successo (test Python su fixture sintetiche, API pubbliche, test ingresso Ruth, test mappe e tools/maps.py --check). Collaudo personale PY-MANUAL-001 positivo sul commit ef4654af8ab106ef194af254d9f3884e5cabb6b0 il 16 ottobre 2026; nessuna verifica GPU o backend nativi.

## Attivita locali

1. `CORE-001` [complete] API comune dell'ispezione Python implementata, testata e integrata in main; altra orchestrazione resta pianificata.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
