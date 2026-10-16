# NVIDIA / hardware/gpu/nvidia

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [NVIDIA](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Provider NVIDIA per inventario, senza certificare backend CUDA.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: not_run; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: candidato read-only con nvidia-smi a parametri fissi e timeout; fixture sintetiche per modello/VRAM/multi-GPU e fallimenti, CI da eseguire. GPU AMD/Intel, driver, CUDA e inferenza esclusi; HW GPU manuale da confermare.

## Attivita locali

1. `NVIDIA-001` [active] Provider nvidia-smi CSV, timeout e report generici senza identificatori persistenti; test e CI da verificare.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
