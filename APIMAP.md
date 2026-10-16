# Ruth / .

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ROOT](NODE.yaml). Ripresa: [START_HERE](START_HERE.md).

## Contratto dichiarato

- `runtime.inspect_environment(project_root: Path) -> PythonEnvironment [integrata in main; ispezione Python sperimentale]`

- `main(argv: Sequence[str] = ()) -> int [sequenza PY->DEP candidata; modalita mirate disponibili]`

Dipendenze: nessuna.

## Contratti delegati

- [Contratti e orchestrazione](runtime/APIMAP.md)
- [Ambiente](environment/APIMAP.md)
- [Hardware](hardware/APIMAP.md)
- [Modelli](models/APIMAP.md)
- [Inferenza](inference/APIMAP.md)
- [Profilazione](profiling/APIMAP.md)
- [Chat](chat/APIMAP.md)
