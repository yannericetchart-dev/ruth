# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

## Contratto dichiarato

- `inspect_header(model_id: str, *, project_root: Path | None = None) -> GGUFHeader [solo id da catalogo, integrata in main]`
- `inspect_headers(*, project_root: Path | None = None) -> GGUFReport [soli GGUF dichiarati, integrata in main]`
- `inspect_metadata(model_id: str, *, project_root: Path | None = None) -> GGUFMetadata [catalog only, CI verified and integrated]`
- `inspect_all_metadata(*, project_root: Path | None = None) -> GGUFMetadataReport [metadata only, CI verified and integrated]`
- `inspect_tensor_index(model_id: str, *, project_root: Path | None = None) -> GGUFTensorIndex [catalog only, CI verified and integrated]`
- `inspect_all_tensor_indexes(*, project_root: Path | None = None) -> GGUFTensorIndexReport [CI verified and integrated]`
- `inspect_block_layout(model_id: str, *, project_root: Path | None = None) -> GGUFBlockLayout [CAT-only, CI verified and integrated]`
- `inspect_all_block_layouts(*, project_root: Path | None = None) -> GGUFBlockLayoutReport [CI verified and integrated]`
- `inspect_kblock_layout(model_id: str, *, project_root: Path | None = None) -> GGUFKBlockLayout [CAT-only, CI verified and integrated]`
- `inspect_all_kblock_layouts(*, project_root: Path | None = None) -> GGUFKBlockReport [CI verified and integrated]`
- `verify(path) -> VerificationReport [proposta non implementata]`

- `run_fixture_pipeline() -> GGUFIndirectFixtureReport [MIG-012 candidate]`

Dipendenze: nessuna.

