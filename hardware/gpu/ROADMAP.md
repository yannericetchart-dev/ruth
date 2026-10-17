# GPU / hardware/gpu

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GPU](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Inventario GPU tramite provider vendor.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope, temporary_branch. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI runs 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici e 258 mappe); PR #30-#33 integrate. GPU-MANUAL-001 passato il 16 ottobre 2026 su c344c71013cfa05da4f49f3204900f4bdc89b8ea: cinque passaggi PY->DEP->DIAG->HW->NVIDIA, stato NVIDIA available e riepilogo positivo. Non certifica CUDA, driver, backend o inferenza; dati host esclusi.

## Attivita locali

1. `GPU-001` [complete] Provider NVIDIA aggregato e testato con report immutabili, PR #30-#33 integrate fino a main.
2. `GPU-SCOPE-001` [complete] Rilevamento NVIDIA tramite nvidia-smi, modello, VRAM totale, errori/timeout e test approvati dall'utente il 16 ottobre 2026.
3. `GPU-MANUAL-001` [complete] Il 16 ottobre 2026 l'utente ha effettuato fast-forward della copia main, verificato HEAD c344c71013cfa05da4f49f3204900f4bdc89b8ea ed eseguito main.py: sequenza di cinque fasi completata, inventario NVIDIA available e riepilogo positivo; nessun dato host in Git.
4. `GPU-PROMOTE-001` [complete] PR #30 in hardware/gpu/nvidia/main (merge 21c17032af358120b7cf0d56246a9c135f2833f7).
5. `GPU-PROMOTE-002` [complete] PR #31 in hardware/gpu/main (merge 0010c43b75c1f28c18b6c83dba0c913f384a40c2).
6. `GPU-PROMOTE-003` [complete] PR #32 in hardware/main (merge a7c5e63ea460c00d9ff5c73913af3835c182b2e2).
7. `GPU-PROMOTE-004` [complete] PR #33 in main (merge 54a02cf37262a78f5a1ed0aa84cd6d210060a14c).

## Componenti (ordine locale, ID stabili)

1. [NVIDIA: NVIDIA](nvidia/ROADMAP.md) — active; complete; passed.
2. [AMD: AMD](amd/ROADMAP.md) — inactive; skeleton; not_run.
3. [INTEL: Intel GPU](intel/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
