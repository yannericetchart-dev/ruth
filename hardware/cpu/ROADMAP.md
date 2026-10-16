# CPU / hardware/cpu

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CPU](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Inventario CPU portabile.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: HW-001/CPU-001: CI GitHub runs 38913317440, 38913438829, 38913533057 passed (74 test sintetici e 249 mappe); PR #25, #26, #27 integrate. Collaudo personale HW-MANUAL-001 passed il 16 ottobre 2026 sul commit 59b26fd6b427e6756ce0285fee60590cb077ebf1, con sequenza PY->DEP->DIAG->HW e report CPU/RAM disponibili; non certifica GPU, CUDA, backend o inferenza.

## Attivita locali

1. `CPU-001` [complete] Rilevamento portabile CPU implementato e verificato con fixture sintetiche; PR #25 in hardware/cpu/main, PR #26 e #27 verso main.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
