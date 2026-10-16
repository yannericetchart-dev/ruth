# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `inspect(project_root: Path) -> PythonEnvironment [integrata in main]`
- `plan(requirements: Iterable[str], *, project_root: Path | None = None) -> DependencyPlan [implementata, verificata e integrata in main]`
- `PythonEnvironment: project_root: str; interpreter: str; python_version: tuple[int, ...]; interpreter_in_project_venv: bool; venv_status: str; reasons: tuple[str, ...] [contratto immutabile della PR #2]`
- `PythonEnvironment.to_dict() -> dict [implementata in main; dati transitori, salvataggi solo locali]`
- `diagnostics.inspect(python_report: PythonEnvironment, dependencies: DependencyPlan | None = None) -> DiagnosticReport [candidata in DIAG-001]`

Dipendenze: nessuna.

## Contratti delegati

- [Python](python/APIMAP.md)
- [Dipendenze](dependencies/APIMAP.md)
- [Diagnostica](diagnostics/APIMAP.md)
