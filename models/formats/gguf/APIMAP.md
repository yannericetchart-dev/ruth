# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

## Contratto dichiarato

- `inspect_header(model_id: str, *, project_root: Path | None = None) -> GGUFHeader [solo id dal catalogo; candidata]`
- `inspect_headers(*, project_root: Path | None = None) -> GGUFReport [soli GGUF dichiarati; candidata]`
- `verify(path) -> VerificationReport [proposta non implementata]`

Dipendenze: nessuna.

