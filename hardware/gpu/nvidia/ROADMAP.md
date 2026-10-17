# NVIDIA / hardware/gpu/nvidia

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [NVIDIA](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Provider NVIDIA per inventario, senza certificare backend CUDA.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope, archived_branch. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI runs 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici e 258 mappe); PR #30-#33 integrate. GPU-MANUAL-001 passato il 16 ottobre 2026 su c344c71013cfa05da4f49f3204900f4bdc89b8ea: cinque passaggi PY->DEP->DIAG->HW->NVIDIA, stato NVIDIA available e riepilogo positivo. Non certifica CUDA, driver, backend o inferenza; dati host esclusi.

## Attivita locali

1. `NVIDIA-001` [complete] Provider NVIDIA read-only via nvidia-smi verificato con fixture sintetiche in CI, integrato in main tramite PR #33 e collaudato personalmente nel quinto passaggio GPU-MANUAL-001.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
