# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `inspect(project_root: Path) -> PythonEnvironment [implementata nella PR #3; non integrata in main]`
- `plan(requirements) -> EnvironmentPlan [proposta]`
- `PythonEnvironment: project_root: str; interpreter: str; python_version: tuple[int, ...]; interpreter_in_project_venv: bool; venv_status: str; reasons: tuple[str, ...] [contratto immutabile della PR #2]`
- `PythonEnvironment.to_dict() -> dict [implementata nella PR #3; dati transitori, salvataggi solo locali]`

Dipendenze: nessuna.

## Contratti delegati

- [Python](python/APIMAP.md)
- [Dipendenze](dependencies/APIMAP.md)
- [Diagnostica](diagnostics/APIMAP.md)
