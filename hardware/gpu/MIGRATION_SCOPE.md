# GPU-SCOPE-001 — NVIDIA (perimetro approvato)

Approvazione dell'utente: 16 ottobre 2026; ref iniziale main
`ae492a1be3b12c0ffd6f9987d0d959aceb4a9ad4`.
Prima migrazione GPU, **solo NVIDIA**. AMD e Intel restano inactive/future
e nessuna API dei loro provider e' implementata.

## API, ingressi e risultati

- `hardware.gpu.nvidia.detect() -> GPUReport`: esegue soltanto il
  comando gia' installato `nvidia-smi --query-gpu=name,memory.total
  --format=csv,noheader,nounits`, senza shell, con timeout di cinque
  secondi; non tenta installazioni, driver, backend o CUDA.
- `hardware.gpu.detect() -> GPUReport`: interfaccia neutra che in
  GPU-SCOPE-001 delega esclusivamente al provider NVIDIA.
- `GPUReport(devices, status)` immutabile: numero `count`, GPU
  `GPUDevice(vendor, model, total_vram_bytes, status)`, `to_dict`
  transitorio. Modello e VRAM totale (MiB da utility, convertiti in
  byte) possono essere sconosciuti. Nessun identificatore persistente
  (seriale/UUID/PCI ID) o log stderr viene raccolto.
- Stati: `available`, `partial`, `none_reported` (query senza righe),
  `unavailable` (utility assente), `timeout`, `inspection_failed`,
  `unknown` (output non interpretabile). `none_reported` non prova
  assenza fisica GPU. Driver non funzionante non e' prova di GPU assente.
- `main.py` default continua PY -> DEP -> DIAG -> HW CPU/RAM ->
  NVIDIA. Il nuovo passaggio e' indipendente dai controlli di
  conformita Python/DEP e non modifica il codice uscita precedente.
  `main.py dependencies ...` resta mirato e non interroga GPU.

## Limiti, protezioni, prove

- Output della utility interpretato entro limiti (righe e lunghezza),
  nominativi filtrati per caratteri sicuri e senza echo di output
  non conforme, stderr o eccezioni contenenti dati host. Nessuna GPU
  inferita se utility indisponibile; GPU AMD/Intel non interrogate.
- Non verificare VRAM libera, compatibilita CUDA, driver richiesti,
  llama.cpp, inferenza, benchmark, quantizzazione o installabilita'.
  Report reale transitorio soltanto sulla macchina dell'utente
  (DATA-001), mai versionato.
- CI con fixture sintetiche: una o piu GPU, assente nvidia-smi,
  errori driver/comando, timeout, righe parziali/malformate, output
  troppo grande, codici stati, regressioni CPU/RAM, ordine CLI e
  mappe. Promozioni:
  `hardware/gpu/nvidia/work/GPU-001` ->
  `hardware/gpu/nvidia/main` -> `hardware/gpu/main` ->
  `hardware/main` -> `main`.
- Dopo merge: collaudo personale `GPU-MANUAL-001` sul commit main
  esatto, via `.venv/bin/python main.py` e conferma esito positivo
  dell'avvio completo. MIG-FLOW-002 blocca il componente successivo
  fino ad accettazione.

**Stato della candidatura:** implementata nel branch di lavoro,
non ancora verificata dalla CI ne' integrata in main.
