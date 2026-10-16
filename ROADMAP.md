# Ruth / .

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ROOT](NODE.yaml). Ripresa: [START_HERE](START_HERE.md).

Coordinare i sottosistemi tramite API comuni.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: HW-MANUAL-001.

Ambito verifica: HW-001/CPU-001: GitHub CI runs 38913317440, 38913438829, 38913533057 success; 14 PY, 12 DEP, 13 DIAG, 8 CPU, 12 HW/RAM, 15 root/CLI; 249 mappe coerenti. PR #25 foglia, #26 hardware/main, #27 main. Test solo sintetici, HW-MANUAL-001 in attesa. GPU, CUDA, backend e installazioni esclusi.

## Attivita locali

1. `SETUP-001` [complete] Configurazione e collaudo indipendente dei client.
2. `STRUCT-001` [complete] Scaffold ricorsivo senza migrazione applicativa.
3. `MIG-001` [complete] Prima porzione PY-001 integrata in main (PR #8) e accettata nel collaudo PY-MANUAL-001 sul commit ef4654af8ab106ef194af254d9f3884e5cabb6b0; prossima porzione soggetta a nuova approvazione funzionale.
4. `MIG-002` [complete] DEP-001 integrata in main e accettata dall'utente sul commit e197cc86829acf52e326a712331cc7ff6cfef6b5; manual DEP-MANUAL-001 concluso.
5. `MAIN-FLOW-001` [complete] Sequenza automatica PY->DEP (pip minimo) verificata su GitHub e integrata in main tramite PR #17, e605a02bdd72441fd4e79d36a943109ee3ba5c57.
6. `MAIN-MANUAL-001` [complete] Utente: avvio sequenziale di main.py dal checkout main pulito, prima Python poi controllo pip, output conforme; commit 3f08cbad9ec06f814031bd1cbcbdc25553bd82ae verificato il 16 ottobre 2026.
7. `MIG-003` [complete] DIAG-001 implementata, verificata e integrata via PR #22; DIAG-MANUAL-001 superato sul commit b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb (sequenza PY->DEP->DIAG). Prossimo componente soltanto dopo nuovo perimetro approvato.
8. `MIG-004` [blocked] HW-SCOPE-001: CPU/RAM implementati, testati e integrati in main via PR #27; collaudo personale HW-MANUAL-001 ancora in attesa prima di una nuova porzione.

## Componenti (ordine locale, ID stabili)

1. [CORE: Contratti e orchestrazione](runtime/ROADMAP.md) — active; partial; passed.
2. [ENV: Ambiente](environment/ROADMAP.md) — active; partial; passed.
3. [HW: Hardware](hardware/ROADMAP.md) — active; complete; passed.
4. [MODEL: Modelli](models/ROADMAP.md) — inactive; skeleton; not_run.
5. [INF: Inferenza](inference/ROADMAP.md) — inactive; skeleton; not_run.
6. [PROF: Profilazione](profiling/ROADMAP.md) — inactive; skeleton; not_run.
7. [CHAT: Chat](chat/ROADMAP.md) — inactive; skeleton; not_run.

