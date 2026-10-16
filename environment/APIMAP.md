# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `inspect(project_root: Path) -> PythonEnvironment [implementata in environment/python/main; non promossa ai padri]`
- `plan(requirements) -> EnvironmentPlan [proposta]`
- `PythonEnvironment: project_root: str; interpreter: str; python_version: tuple[int, ...]; interpreter_in_project_venv: bool; venv_status: str; reasons: tuple[str, ...] [dataclass immutabile]`
- `PythonEnvironment.to_dict() -> dict [dati transitori, salvataggi solo locali]`

Dipendenze: nessuna.

## Contratti delegati

- [Python](python/APIMAP.md)
- [Dipendenze](dependencies/APIMAP.md)
- [Diagnostica](diagnostics/APIMAP.md)
