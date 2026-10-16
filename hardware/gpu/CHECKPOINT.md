# GPU: checkpoint GPU-001 NVIDIA

**GPU-SCOPE-001 approvato, implementato, testato e integrato in main**,
solo NVIDIA. AMD e Intel rimangono inattivi e non migrati.

- Provider `nvidia-smi` senza shell, parametri fissi, timeout cinque
  secondi e parsing CSV bounded, senza identificativi persistenti.
- Report `GPUDevice` e `GPUReport` (vendor, modello, VRAM totale,
  conteggio e stato), transitori, non certificano driver/CUDA.
- Promozioni: PR #30 foglia (`21c17032af358120b7cf0d56246a9c135f2833f7`),
  PR #31 GPU (`0010c43b75c1f28c18b6c83dba0c913f384a40c2`),
  PR #32 HW (`a7c5e63ea460c00d9ff5c73913af3835c182b2e2`),
  PR #33 main (`54a02cf37262a78f5a1ed0aa84cd6d210060a14c`).
- GitHub CI run 38918374871, 38918650589, 38918759116,
  38918867451: success; 92 fixture sintetiche e 258 mappe.
- **GPU-MANUAL-001: active**, da eseguire sul clone main pulito dopo
  sincronizzazione e conferma SHA; nessuna osservazione GPU reale
  pubblicata. Nessun prossimo componente prima del collaudo.
- DATA-001, REMOTE-001, MIG-FLOW-001/002 vigenti.
