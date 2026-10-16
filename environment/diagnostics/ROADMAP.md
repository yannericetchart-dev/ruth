# Diagnostica / environment/diagnostics

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [DIAG](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Spiegare problemi di ambiente senza alterarlo.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DIAG-001: GitHub Actions 38908192677, 38908269925, 38908370200 superate (14 PY, 12 DEP, 13 DIAG, 15 test root/CLI, mappe coerenti). PR #20 figlio, #21 environment/main e #22 main. Solo analisi sintetiche di report; collaudo DIAG-MANUAL-001 pendente; nessuna diagnostica GPU/CUDA.

## Attivita locali

1. `DIAG-001` [complete] Contratto immutabile, diagnostica dei report e passaggio PY->DEP->DIAG verificati in CI e integrati in main tramite PR #22.
2. `DIAG-SCOPE-001` [complete] Funzioni, ingressi, risultati, effetti e limiti approvati esplicitamente dall'utente il 16 ottobre 2026.
3. `DIAG-MANUAL-001` [active] Attende sincronizzazione non distruttiva della copia main e conferma personale dell'ordine PY->DEP->DIAG e del commit effettivamente collaudato.
4. `DIAG-PROMOTE-001` [complete] PR #20 integrata in environment/diagnostics/main, merge ec7eba74bc71e62ea8562fa51d62295f5f3c01b8.
5. `DIAG-PROMOTE-002` [complete] PR #21 integrata in environment/main, merge 69ae0821cc0043bd76254bc46dff12b40d0d7b30.
6. `DIAG-PROMOTE-003` [complete] PR #22 integrata in main, merge dfdb41e981f32ee233320f77998aadab55ce4d7b.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
