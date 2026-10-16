# GPU / hardware/gpu

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GPU](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Inventario GPU tramite provider vendor.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: not_run; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: candidato read-only con nvidia-smi a parametri fissi e timeout; fixture sintetiche per modello/VRAM/multi-GPU e fallimenti, CI da eseguire. GPU AMD/Intel, driver, CUDA e inferenza esclusi; HW GPU manuale da confermare.

## Attivita locali

1. `GPU-001` [active] Aggregazione del provider NVIDIA, report GPUDevice/GPUReport portabili; test, CI e promozioni pendenti.
2. `GPU-SCOPE-001` [complete] Rilevamento NVIDIA tramite nvidia-smi, modello, VRAM totale, errori/timeout e test approvati dall'utente il 16 ottobre 2026.
3. `GPU-MANUAL-001` [planned] Dopo merge main: avvio Ruth da copia main pulita e conferma dei cinque passaggi, esito GPU e SHA; dati host non in Git.

## Componenti (ordine locale, ID stabili)

1. [NVIDIA: NVIDIA](nvidia/ROADMAP.md) — active; complete; not_run.
2. [AMD: AMD](amd/ROADMAP.md) — inactive; skeleton; not_run.
3. [INTEL: Intel GPU](intel/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
