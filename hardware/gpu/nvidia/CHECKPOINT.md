# NVIDIA: checkpoint NVIDIA-001

**Implementato, verificato e integrato in main** per l'ambito
GPU-SCOPE-001 autorizzato il 16 ottobre 2026.

- Fonte: interrogazione read-only `nvidia-smi` su modello e VRAM
  totale, nessun UUID, seriale o log stderr usato.
- API `hardware.gpu.nvidia.detect() -> GPUReport` tramite contratti
  pubblici `hardware.gpu`; AMD/Intel non interpellati.
- CI 38918374871, 38918650589, 38918759116 e 38918867451
  superate: 16 test provider, suite PY/DEP/DIAG/HW/CLI e mappe.
- PR #30, #31, #32 e #33 integrate fino a main, merge finale
  `54a02cf37262a78f5a1ed0aa84cd6d210060a14c`.
- GPU-MANUAL-001: **pending**. Nessuna prova effettiva sulla
  macchina utente ancora confermata, nessun dato host salvato.
