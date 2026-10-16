# GPU / hardware/gpu

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GPU](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

## Contratto dichiarato

- `detect() -> GPUReport [solo NVIDIA, verificata e integrata in main; AMD/Intel inattive]`

Dipendenze: nessuna.

## Contratti delegati

- [NVIDIA](nvidia/APIMAP.md)
- [AMD](amd/APIMAP.md)
- [Intel GPU](intel/APIMAP.md)
