# GPU / hardware/gpu

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GPU](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Inventario GPU tramite provider vendor.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GPU-001 NVIDIA: CI GitHub 38918374871, 38918650589, 38918759116, 38918867451 success (92 test sintetici, 258 mappe); promozioni PR #30 NVIDIA, #31 GPU, #32 hardware e #33 main. Driver, CUDA, AMD/Intel e compatibilita' inferenza non verificati. GPU-MANUAL-001 ancora pendente.

## Attivita locali

1. `GPU-001` [complete] Provider NVIDIA aggregato e testato con report immutabili, PR #30-#33 integrate fino a main.
2. `GPU-SCOPE-001` [complete] Rilevamento NVIDIA tramite nvidia-smi, modello, VRAM totale, errori/timeout e test approvati dall'utente il 16 ottobre 2026.
3. `GPU-MANUAL-001` [active] Da eseguire dopo sync fast-forward del checkout pulito: main.py senza argomenti, verifica quinta fase NVIDIA e commit Git; inviare solo esito generale, non dati host.
4. `GPU-PROMOTE-001` [complete] PR #30 in hardware/gpu/nvidia/main (merge 21c17032af358120b7cf0d56246a9c135f2833f7).
5. `GPU-PROMOTE-002` [complete] PR #31 in hardware/gpu/main (merge 0010c43b75c1f28c18b6c83dba0c913f384a40c2).
6. `GPU-PROMOTE-003` [complete] PR #32 in hardware/main (merge a7c5e63ea460c00d9ff5c73913af3835c182b2e2).
7. `GPU-PROMOTE-004` [complete] PR #33 in main (merge 54a02cf37262a78f5a1ed0aa84cd6d210060a14c).

## Componenti (ordine locale, ID stabili)

1. [NVIDIA: NVIDIA](nvidia/ROADMAP.md) — active; complete; passed.
2. [AMD: AMD](amd/ROADMAP.md) — inactive; skeleton; not_run.
3. [INTEL: Intel GPU](intel/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
