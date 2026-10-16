# Contratti e orchestrazione / runtime

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CORE](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `inspect_environment(project_root: Path) -> PythonEnvironment [implementata nel branch di lavoro]`
- `PythonEnvironment [contratto comune importato da environment]`

Dipendenze: [environment](../environment/APIMAP.md), [hardware](../hardware/APIMAP.md), [models](../models/APIMAP.md), [inference](../inference/APIMAP.md), [profiling](../profiling/APIMAP.md), [chat](../chat/APIMAP.md).

