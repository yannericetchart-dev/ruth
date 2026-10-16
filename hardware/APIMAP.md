# Hardware / hardware

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [HW](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `detect() -> HardwareReport [CPU e RAM, candidata read-only]`
- `resources(device: DeviceInfo | str = 'system') -> ResourceReport [RAM totale fisica, candidata read-only]`

Dipendenze: nessuna.

## Contratti delegati

- [CPU](cpu/APIMAP.md)
- [GPU](gpu/APIMAP.md)
