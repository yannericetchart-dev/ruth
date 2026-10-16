# Formati / models/formats

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [FORMAT](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

## Contratto dichiarato

- `inspect_headers(*, project_root: Path | None = None) -> GGUFReport [GGUF-001 verificata e integrata]`
- `inspect_all_metadata(*, project_root: Path | None = None) -> GGUFMetadataReport [metadata bounded candidata]`

Dipendenze: nessuna.

## Contratti delegati

- [GGUF](gguf/APIMAP.md)
