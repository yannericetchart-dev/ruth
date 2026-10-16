# Dipendenze / environment/dependencies

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [DEP](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Risolvere e verificare dipendenze generiche senza lock di macchina.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DEP-001: GitHub Actions 38895348468 (14 Python, 12 dipendenze, 13 root/CLI, 237 mappe), e 38895479549 (promozione padre): success su fixture sintetiche; nessuna verifica nativa, sicurezza o installabilita' certificata. DEP-MANUAL-001 ancora da svolgere dopo main.

## Attivita locali

1. `DEP-001` [complete] Ispezione metadata, gate venv e CLI verificati da CI, promossi da work a dependencies/main PR #12 e environment/main PR #13.
2. `DEP-SCOPE-001` [complete] Perimetro funzionale e sequenza dipendenze prima di hardware approvati dall'utente il 16 ottobre 2026.
3. `DEP-MANUAL-001` [planned] Dopo integrazione main: eseguire personalmente .venv/bin/python main.py dependencies pip e confermare esito con commit Git.
4. `DEP-PROMOTE-001` [complete] PR #12 promossa in environment/dependencies/main al merge 4af2050e5fc59477f9ab854173bf915b7933d186.
5. `DEP-PROMOTE-002` [complete] PR #13 promossa in environment/main al merge c89c02ac0afa828b2c53ebd922a04331069ba527.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
