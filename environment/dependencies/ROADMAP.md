# Dipendenze / environment/dependencies

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [DEP](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Risolvere e verificare dipendenze generiche senza lock di macchina.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope, temporary_branch. Blocchi: nessuno registrato.

Ambito verifica: MAIN-FLOW-001: CI GitHub run 38900998138 e 38901495978 superate per test Python/dipendenze, ordine e gate dell'ingresso; integrazione PR #17 in main. MAIN-MANUAL-001 superato su revisione 3f08cbad9ec06f814031bd1cbcbdc25553bd82ae (16 ottobre 2026), sequenza PY->DEP con pip, senza certificare runtime/hardware.

## Attivita locali

1. `DEP-001` [complete] Ispezione metadata con gate .venv e CLI integrata in main (PR #15), verificata su CI remota.
2. `DEP-SCOPE-001` [complete] Perimetro funzionale e sequenza dipendenze prima di hardware approvati dall'utente il 16 ottobre 2026.
3. `DEP-MANUAL-001` [complete] Esecuzione personale .venv/bin/python main.py dependencies pip, esito present_unverified, commit e197cc86829acf52e326a712331cc7ff6cfef6b5 verificato, confermata dall'utente il 16 ottobre 2026.
4. `DEP-PROMOTE-001` [complete] PR #12 promossa in environment/dependencies/main al merge 4af2050e5fc59477f9ab854173bf915b7933d186.
5. `DEP-PROMOTE-002` [complete] PR #13 promossa in environment/main al merge c89c02ac0afa828b2c53ebd922a04331069ba527.
6. `DEP-PROMOTE-003` [complete] PR #15 integrata in main al merge a542f9ee9f6d09a0496bf756615d2295a7f59ce0.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
