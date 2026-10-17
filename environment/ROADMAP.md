# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Ispezionare Python e pianificare dipendenze senza installazioni implicite.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope, archived_branch. Blocchi: nessuno registrato.

Ambito verifica: DIAG-001: test sintetici GitHub Actions 38908192677, 38908269925, 38908370200 superati (14 PY, 12 DEP, 13 DIAG, 15 root/CLI; mappe coerenti), integrazione PR #20-#22. DIAG-MANUAL-001 superato il 16 ottobre 2026 sul commit b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb: ordine PY->DEP->DIAG e output di riepilogo conformi. Nessuna certificazione hardware, CUDA o runtime.

## Attivita locali

1. `ENV-001` [complete] Porzione PY-001 testata e promossa da environment/python/main a environment/main (PR #3) e main (PR #8).
2. `ENV-002` [complete] DEP-001 verificata e promossa in environment/main via PR #13 e in main via PR #15.
3. `ENV-003` [complete] Diagnostica DIAG-001 verificata e promossa da foglia PR #20 a environment/main PR #21 e main PR #22.

## Componenti (ordine locale, ID stabili)

1. [PY: Python](python/ROADMAP.md) — active; partial; passed.
2. [DEP: Dipendenze](dependencies/ROADMAP.md) — active; complete; passed.
3. [DIAG: Diagnostica](diagnostics/ROADMAP.md) — active; complete; passed.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
