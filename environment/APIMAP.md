# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `inspect(project_root) -> EnvironmentReport [proposta]`
- `plan(requirements) -> EnvironmentPlan [proposta]`

Dipendenze: nessuna.

## Contratti delegati

- [Python](python/APIMAP.md)
- [Dipendenze](dependencies/APIMAP.md)
- [Diagnostica](diagnostics/APIMAP.md)
