# Ruth / .

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ROOT](NODE.yaml). Ripresa: [START_HERE](START_HERE.md).

Coordinare i sottosistemi tramite API comuni.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: stale; integrazione: partial.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DEP-001: validazione di nomi/versioni, report dei metadati tramite lookup sintetico e controllo del blocco Python; test su CI da eseguire. PY-001 gia' verificato separatamente.

## Attivita locali

1. `SETUP-001` [complete] Configurazione e collaudo indipendente dei client.
2. `STRUCT-001` [complete] Scaffold ricorsivo senza migrazione applicativa.
3. `MIG-001` [complete] Prima porzione PY-001 integrata in main (PR #8) e accettata nel collaudo PY-MANUAL-001 sul commit ef4654af8ab106ef194af254d9f3884e5cabb6b0; prossima porzione soggetta a nuova approvazione funzionale.
4. `MIG-002` [active] DEP-SCOPE-001 approvato; sviluppo remoto della verifica dipendenze generiche e CLI, test e promozioni in corso.

## Componenti (ordine locale, ID stabili)

1. [CORE: Contratti e orchestrazione](runtime/ROADMAP.md) — active; partial; passed.
2. [ENV: Ambiente](environment/ROADMAP.md) — active; partial; stale.
3. [HW: Hardware](hardware/ROADMAP.md) — inactive; skeleton; not_run.
4. [MODEL: Modelli](models/ROADMAP.md) — inactive; skeleton; not_run.
5. [INF: Inferenza](inference/ROADMAP.md) — inactive; skeleton; not_run.
6. [PROF: Profilazione](profiling/ROADMAP.md) — inactive; skeleton; not_run.
7. [CHAT: Chat](chat/ROADMAP.md) — inactive; skeleton; not_run.

