# Formati / models/formats

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [FORMAT](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

## Contratto dichiarato

- `inspect_headers(*, project_root: Path | None = None) -> GGUFReport [GGUF-001 verificata e integrata]`
- `inspect_all_metadata(*, project_root: Path | None = None) -> GGUFMetadataReport [metadata bounded CI verified and integrated]`
- `inspect_all_tensor_indexes(*, project_root: Path | None = None) -> GGUFTensorIndexReport [CI verified and integrated]`
- `inspect_all_block_layouts(*, project_root: Path | None = None) -> GGUFBlockLayoutReport [CI verified and integrated]`

Dipendenze: nessuna.

## Contratti delegati

- [GGUF](gguf/APIMAP.md)
