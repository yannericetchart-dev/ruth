# GPU: checkpoint GPU-001 NVIDIA

**GPU-SCOPE-001 approvata, implementata, verificata, integrata in main
e collaudata personalmente dall'utente** il 16 ottobre 2026.
AMD e Intel rimangono inattive, senza migrazione implicita.

- Provider read-only `nvidia-smi` senza shell, con timeout e
  parametri fissi, nessun UUID, seriale o stderr pubblicato.
- Report `GPUDevice` e `GPUReport` (conteggio, modello, VRAM totale
  e stato) transitori, non certificano driver o compatibilita' CUDA.
- Promozioni: PR #30 -> NVIDIA, PR #31 -> GPU, PR #32 -> hardware,
  PR #33 -> main; merge root
  `54a02cf37262a78f5a1ed0aa84cd6d210060a14c`.
  PR #34 checkpoint finale tecnico, commit `c344c71013cfa05da4f49f3204900f4bdc89b8ea`.
- CI GitHub: 38918374871, 38918650589, 38918759116, 38918867451
  e 38919460341 success (92 test sintetici, 258 mappe).
- **GPU-MANUAL-001 complete**: sul commit `c344c71013cfa05da4f49f3204900f4bdc89b8ea` verificato
  tramite Git dopo fast-forward, l'utente ha avviato personalmente
  Ruth confermando tutte e cinque le fasi nell'ordine; il provider
  NVIDIA ha restituito stato `available` e il flusso e' terminato
  positivamente.
- Nessun dato host/versione/modello/VRAM osservato viene registrato
  nel repository (DATA-001). I prossimi provider richiedono uno
  scope separato approvato (MIG-FLOW-001).
