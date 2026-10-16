# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

## Contratto dichiarato

- `inspect_header(model_id: str, *, project_root: Path | None = None) -> GGUFHeader [solo id da catalogo, integrata in main]`
- `inspect_headers(*, project_root: Path | None = None) -> GGUFReport [soli GGUF dichiarati, integrata in main]`
- `inspect_metadata(model_id: str, *, project_root: Path | None = None) -> GGUFMetadata [catalog only, CI verified and integrated]`
- `inspect_all_metadata(*, project_root: Path | None = None) -> GGUFMetadataReport [metadata only, CI verified and integrated]`
- `verify(path) -> VerificationReport [proposta non implementata]`

Dipendenze: nessuna.

