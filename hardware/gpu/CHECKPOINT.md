# GPU: checkpoint GPU-001

- Scope NVIDIA GPU-SCOPE-001 approvato il 16 ottobre 2026.
- Implementato provider nvidia-smi a parametri fissi, report
  GPUDevice/GPUReport, errori/timeout generici, senza CUDA.
- Branch di lavoro `hardware/gpu/nvidia/work/GPU-001`;
  integrazione a main e test CI ancora non eseguiti.
- `main.py` candidato: PY -> DEP -> DIAG -> HW CPU/RAM -> NVIDIA.
- AMD e Intel restano inattivi. Dati reali GPU/host esclusi da Git.
- GPU-MANUAL-001 non eseguito; prossimo passo: mappe, test CI,
  promozioni e collaudo personale post-merge.
