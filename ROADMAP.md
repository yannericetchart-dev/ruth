# Ruth / .

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ROOT](NODE.yaml). Ripresa: [START_HERE](START_HERE.md).

Coordinare i sottosistemi tramite API comuni.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: GGUF-BLOCK-MANUAL-001.

Ambito verifica: CAT-001: CI GitHub 38928909391, 38929028829, 38929118836 success (116 test sintetici inclusi 21 catalogo e 20 CLI, 264 mappe). PR #36 models/catalog/main, #37 models/main e #38 main. Manifest locale read-only, file solo dichiarati, niente download, lettura GGUF, inferenza o dati reali Git. CAT-MANUAL-001 confermato positivo sul commit 90e57f106fb664b0257775260044125ea85228e2. GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference. GGUF-META-001: CI success runs 38940033886, 38940265778, 38940360999, 38940625555; PR #47 gguf, #48 formats, #49 models, #50 main (a630c83f6c62a229e18fe5e64f0863c19de16018). Catalog-only bounded KV reader, architecture/file_type/context_length optional; eight-step Ruth, synthetic tests and generated maps passed. GGUF-META-MANUAL-001 accepted on d318fd793da514f21ee2e8475304d1a6c883ee94 (catalog absent; no real GGUF tested); no tensors/download/inference. GGUF-INDEX-001: CI success runs 38945221302, 38945375637, 38945463381, 38945568401. PR #53 gguf, #54 formats, #55 Models, #56 main (1d51cfeed2bec83119ebf56ff0cb3840089484cf). Bounded tensor-index structural checks; unknown/quantized sizes unverified; no tensor payload. GGUF-INDEX-MANUAL-001 accepted personally on 09045243e944514d40b9e58c6159b013efa2a196 (catalog not configured; no real GGUF tested). GGUF-BLOCK-001: PR #59 gguf, #60 formats, #61 Models, #62 main (5f896f1bbacaa8c802f66f2a40ee7663cfa40604). CI runs 39022476943, 39022574327, 39022637716, 39022698747 passed (fixture sintetiche, entrypoint and maps); five GGML types 2/3/6/7/8 validated with block extents, no tensor payload; GGUF-BLOCK-MANUAL-001 pending.

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
11. `MIG-007` [complete] GGUF-SCOPE-001: bounded header reader verified by CI and merged via PR #44; GGUF-MANUAL-001 personally accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured). New scope requires approval.
12. `MIG-008` [complete] GGUF-META-SCOPE-001 implemented, CI verified and integrated through PR #50; GGUF-META-MANUAL-001 accepted personally on d318fd793da514f21ee2e8475304d1a6c883ee94 (catalog not configured). Next scope must be approved.
13. `MIG-009` [complete] GGUF-INDEX-SCOPE-001 implemented and CI verified, merged via PR #56; GGUF-INDEX-MANUAL-001 personally accepted on 09045243e944514d40b9e58c6159b013efa2a196 with catalog not_configured. Next functional scope requires explicit approval.
14. `MIG-010` [blocked] GGUF-BLOCK-SCOPE-001 five GGML types implemented, CI passed and integrated via PR #62 in main 5f896f1bbacaa8c802f66f2a40ee7663cfa40604; GGUF-BLOCK-MANUAL-001 pending before next migration.

## Componenti (ordine locale, ID stabili)

1. [CORE: Contratti e orchestrazione](runtime/ROADMAP.md) — active; partial; passed.
2. [ENV: Ambiente](environment/ROADMAP.md) — active; partial; passed.
3. [HW: Hardware](hardware/ROADMAP.md) — active; complete; passed.
4. [MODEL: Modelli](models/ROADMAP.md) — active; partial; passed.
5. [INF: Inferenza](inference/ROADMAP.md) — inactive; skeleton; not_run.
6. [PROF: Profilazione](profiling/ROADMAP.md) — inactive; skeleton; not_run.
7. [CHAT: Chat](chat/ROADMAP.md) — inactive; skeleton; not_run.

