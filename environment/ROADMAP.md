# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Ispezionare Python e pianificare dipendenze senza installazioni implicite.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DEP-001: collaudo manuale positivo sul commit e197cc86829acf52e326a712331cc7ff6cfef6b5 (16 ottobre 2026); controlli sintetici CI precedenti superati. MAIN-FLOW-001: sequenza predefinita PY->DEP candidata in PR, nuovi test da eseguire; test hardware, lock e compatibilita' runtime esclusi.

## Attivita locali

1. `ENV-001` [complete] Porzione PY-001 testata e promossa da environment/python/main a environment/main (PR #3) e main (PR #8).
2. `ENV-002` [complete] DEP-001 verificata e promossa in environment/main via PR #13 e in main via PR #15.

## Componenti (ordine locale, ID stabili)

1. [PY: Python](python/ROADMAP.md) — active; partial; passed.
2. [DEP: Dipendenze](dependencies/ROADMAP.md) — active; complete; passed.
3. [DIAG: Diagnostica](diagnostics/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
