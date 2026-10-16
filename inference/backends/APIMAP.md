# Backend / inference/backends

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [BACKEND](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

## Contratto dichiarato

- `assess(device) -> BackendAssessment [proposta]`

Dipendenze: nessuna.

## Contratti delegati

- [CUDA](cuda/APIMAP.md)
- [Vulkan](vulkan/APIMAP.md)
- [ROCm/HIP](rocm_hip/APIMAP.md)
- [SYCL](sycl/APIMAP.md)
