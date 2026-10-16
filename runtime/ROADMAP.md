# Contratti e orchestrazione / runtime

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CORE](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Esporre i contratti comuni e coordinare i domini.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: CI GitHub Actions run 38900998138: 14 Python, 12 dipendenze, 14 root/CLI (inclusi sequenza e gate), controllo di 237 mappe: success. Merge PR #17 in main e605a02bdd72441fd4e79d36a943109ee3ba5c57; manual MAIN-MANUAL-001 in attesa; nessuna certificazione host/GPU/lock.

## Attivita locali

1. `CORE-001` [complete] API comune dell'ispezione Python implementata, testata e integrata in main; altra orchestrazione resta pianificata.
2. `CORE-FLOW-001` [complete] Orchestrazione sequenziale del punto d'ingresso verificata e integrata in main tramite PR #17; collaudo personale del default ancora pendente.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
