# Contratti e orchestrazione / runtime

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CORE](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Esporre i contratti comuni e coordinare i domini.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: MAIN-FLOW-001: CI GitHub run 38900998138 e 38901495978 superate per test Python/dipendenze, ordine e gate dell'ingresso; integrazione PR #17 in main. MAIN-MANUAL-001 superato su revisione 3f08cbad9ec06f814031bd1cbcbdc25553bd82ae (16 ottobre 2026), sequenza PY->DEP con pip, senza certificare runtime/hardware.

## Attivita locali

1. `CORE-001` [complete] API comune dell'ispezione Python implementata, testata e integrata in main; altra orchestrazione resta pianificata.
2. `CORE-FLOW-001` [complete] Orchestrazione default PY->DEP verificata in CI e integrata via PR #17; collaudo personale MAIN-MANUAL-001 positivo al commit 3f08cbad9ec06f814031bd1cbcbdc25553bd82ae.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
