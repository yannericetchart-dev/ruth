# NVIDIA / hardware/gpu/nvidia

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [NVIDIA](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Provider NVIDIA per inventario, senza certificare backend CUDA.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI GitHub 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici, 258 mappe); promozioni PR #30 NVIDIA, #31 GPU, #32 hardware e #33 main. Driver, CUDA, AMD/Intel e compatibilita' inferenza non verificati. GPU-MANUAL-001 ancora pendente.

## Attivita locali

1. `NVIDIA-001` [complete] Provider NVIDIA read-only via nvidia-smi, fixture sintetiche verificate in CI e integrato in main tramite PR #33.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
