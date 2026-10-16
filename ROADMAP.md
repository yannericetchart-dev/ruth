# Ruth / .

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ROOT](NODE.yaml). Ripresa: [START_HERE](START_HERE.md).

Coordinare i sottosistemi tramite API comuni.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI runs 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici e 258 mappe); PR #30-#33 integrate. GPU-MANUAL-001 passato il 16 ottobre 2026 su c344c71013cfa05da4f49f3204900f4bdc89b8ea: cinque passaggi PY->DEP->DIAG->HW->NVIDIA, stato NVIDIA available e riepilogo positivo. Non certifica CUDA, driver, backend o inferenza; dati host esclusi.

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

## Componenti (ordine locale, ID stabili)

1. [CORE: Contratti e orchestrazione](runtime/ROADMAP.md) — active; partial; passed.
2. [ENV: Ambiente](environment/ROADMAP.md) — active; partial; passed.
3. [HW: Hardware](hardware/ROADMAP.md) — active; complete; passed.
4. [MODEL: Modelli](models/ROADMAP.md) — inactive; skeleton; not_run.
5. [INF: Inferenza](inference/ROADMAP.md) — inactive; skeleton; not_run.
6. [PROF: Profilazione](profiling/ROADMAP.md) — inactive; skeleton; not_run.
7. [CHAT: Chat](chat/ROADMAP.md) — inactive; skeleton; not_run.

