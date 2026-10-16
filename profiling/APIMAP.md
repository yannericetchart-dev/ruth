# Profilazione / profiling

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [PROF](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `plan(request) -> ProfilePlan [proposta]`
- `run(plan) -> ProfileResult [proposta]`
- `record(result, local_store) -> ProfileReference [proposta]`
- `select(request) -> ProfileSelection [proposta]`

Dipendenze: [models](../models/APIMAP.md), [hardware](../hardware/APIMAP.md), [inference](../inference/APIMAP.md).

## Contratti delegati

- [Selezione](selection/APIMAP.md)
- [Memoria](memory/APIMAP.md)
- [Contesto](context/APIMAP.md)
- [Placement](placement/APIMAP.md)
- [Precisione](precision/APIMAP.md)
- [Profili](profiles/APIMAP.md)
