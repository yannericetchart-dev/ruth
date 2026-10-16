# Dipendenze / environment/dependencies

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [DEP](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Risolvere e verificare dipendenze generiche senza lock di macchina.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DEP-001: GitHub Actions 38895348468, 38895479549, 38895836703 e 38895924627, success: 14 test Python, 12 dipendenze, 13 CLI/root e 237 mappe. Test sintetici e metadati soltanto; non verifica installabilita', lock, GPU o backend. DEP-MANUAL-001 in attesa.

## Attivita locali

1. `DEP-001` [complete] Ispezione metadata con gate .venv e CLI integrata in main (PR #15), verificata su CI remota.
2. `DEP-SCOPE-001` [complete] Perimetro funzionale e sequenza dipendenze prima di hardware approvati dall'utente il 16 ottobre 2026.
3. `DEP-MANUAL-001` [active] In attesa di collaudo personale dell'utente dopo sincronizzazione sicura: .venv/bin/python main.py dependencies pip e conferma commit main.
4. `DEP-PROMOTE-001` [complete] PR #12 promossa in environment/dependencies/main al merge 4af2050e5fc59477f9ab854173bf915b7933d186.
5. `DEP-PROMOTE-002` [complete] PR #13 promossa in environment/main al merge c89c02ac0afa828b2c53ebd922a04331069ba527.
6. `DEP-PROMOTE-003` [complete] PR #15 integrata in main al merge a542f9ee9f6d09a0496bf756615d2295a7f59ce0.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
