# Contratti e orchestrazione / runtime

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CORE](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Esporre i contratti comuni e coordinare i domini.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: stale; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DEP-001: collaudo manuale positivo sul commit e197cc86829acf52e326a712331cc7ff6cfef6b5 (16 ottobre 2026); controlli sintetici CI precedenti superati. MAIN-FLOW-001: sequenza predefinita PY->DEP candidata in PR, nuovi test da eseguire; test hardware, lock e compatibilita' runtime esclusi.

## Attivita locali

1. `CORE-001` [complete] API comune dell'ispezione Python implementata, testata e integrata in main; altra orchestrazione resta pianificata.
2. `CORE-FLOW-001` [active] Orchestrazione del punto di ingresso default per controlli integrati ordinati, senza mutare gli host; verifica CI e merge ancora pendenti.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
