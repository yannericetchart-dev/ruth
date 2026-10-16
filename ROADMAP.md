# Ruth / .

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ROOT](NODE.yaml). Ripresa: [START_HERE](START_HERE.md).

Coordinare i sottosistemi tramite API comuni.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: GGUF-MANUAL-001.

Ambito verifica: CAT-001: CI GitHub 38928909391, 38929028829, 38929118836 success (116 test sintetici inclusi 21 catalogo e 20 CLI, 264 mappe). PR #36 models/catalog/main, #37 models/main e #38 main. Manifest locale read-only, file solo dichiarati, niente download, lettura GGUF, inferenza o dati reali Git. CAT-MANUAL-001 confermato positivo sul commit 90e57f106fb664b0257775260044125ea85228e2. GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 pending; no model payload or inference.

## Attivita locali

1. `SETUP-001` [complete] Configurazione e collaudo indipendente dei client.
2. `STRUCT-001` [complete] Scaffold ricorsivo senza migrazione applicativa.
3. `MIG-001` [complete] Prima porzione PY-001 integrata in main (PR #8) e accettata nel collaudo PY-MANUAL-001 sul commit ef4654af8ab106ef194af254d9f3884e5cabb6b0; prossima porzione soggetta a nuova approvazione funzionale.
4. `MIG-002` [complete] DEP-001 integrata in main e accettata dall'utente sul commit e197cc86829acf52e326a712331cc7ff6cfef6b5; manual DEP-MANUAL-001 concluso.
5. `MAIN-FLOW-001` [complete] Sequenza automatica PY->DEP (pip minimo) verificata su GitHub e integrata in main tramite PR #17, e605a02bdd72441fd4e79d36a943109ee3ba5c57.
6. `MAIN-MANUAL-001` [complete] Utente: avvio sequenziale di main.py dal checkout main pulito, prima Python poi controllo pip, output conforme; commit 3f08cbad9ec06f814031bd1cbcbdc25553bd82ae verificato il 16 ottobre 2026.
7. `MIG-003` [complete] DIAG-001 implementata, verificata e integrata via PR #22; DIAG-MANUAL-001 superato sul commit b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb (sequenza PY->DEP->DIAG). Prossimo componente soltanto dopo nuovo perimetro approvato.
8. `MIG-004` [complete] HW-SCOPE-001 CPU/RAM implementata, verificata e integrata via PR #27; collaudo personale HW-MANUAL-001 positivo sul commit 59b26fd6b427e6756ce0285fee60590cb077ebf1 (PY->DEP->DIAG->HW). Prossima porzione richiede nuovo perimetro approvato.
9. `MIG-005` [complete] GPU-SCOPE-001 NVIDIA implementata, verificata e integrata in main via PR #33; GPU-MANUAL-001 confermata positiva sul commit c344c71013cfa05da4f49f3204900f4bdc89b8ea con sequenza PY->DEP->DIAG->HW->NVIDIA. Il prossimo perimetro richiede nuova approvazione.
10. `MIG-006` [complete] CAT-SCOPE-001: catalogo locale read-only implementato, verificato e integrato in main (PR #38); CAT-MANUAL-001 superato personalmente sul commit 90e57f106fb664b0257775260044125ea85228e2. Prossimo perimetro da approvare.
11. `MIG-007` [blocked] GGUF-SCOPE-001 approvato, implementato e verificato in CI, integrato in main via PR #44 (commit 15b106082c4dee37622c61a8afefd248a57464a0); GGUF-MANUAL-001 personale pendente prima della prossima porzione.

## Componenti (ordine locale, ID stabili)

1. [CORE: Contratti e orchestrazione](runtime/ROADMAP.md) — active; partial; passed.
2. [ENV: Ambiente](environment/ROADMAP.md) — active; partial; passed.
3. [HW: Hardware](hardware/ROADMAP.md) — active; complete; passed.
4. [MODEL: Modelli](models/ROADMAP.md) — active; partial; passed.
5. [INF: Inferenza](inference/ROADMAP.md) — inactive; skeleton; not_run.
6. [PROF: Profilazione](profiling/ROADMAP.md) — inactive; skeleton; not_run.
7. [CHAT: Chat](chat/ROADMAP.md) — inactive; skeleton; not_run.

