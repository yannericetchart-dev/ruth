# Contratti e orchestrazione / runtime

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CORE](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `inspect_environment(project_root: Path) -> PythonEnvironment [verificata in environment/main; non ancora integrata in main]`

Dipendenze: [environment](../environment/APIMAP.md), [hardware](../hardware/APIMAP.md), [models](../models/APIMAP.md), [inference](../inference/APIMAP.md), [profiling](../profiling/APIMAP.md), [chat](../chat/APIMAP.md).

