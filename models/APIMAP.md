# Modelli / models

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [MODEL](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `catalog(*, project_root: Path | None = None) -> ModelCatalog [CAT-001 verificata e integrata, manifest locale]`
- `resolve(reference: str, *, project_root: Path | None = None) -> ModelRecord | None [CAT-001 verificata e integrata]`
- `inspect_all_metadata(*, project_root: Path | None = None) -> GGUFMetadataReport [GGUF-META-001 CI verified and integrated]`
- `inspect_all_tensor_indexes(*, project_root: Path | None = None) -> GGUFTensorIndexReport [GGUF-INDEX-001 CI verified and integrated]`
- `acquire(request) -> AcquisitionResult [proposta]`
- `verify(artifact) -> VerificationReport [proposta]`

Dipendenze: nessuna.

## Contratti delegati

- [Catalogo](catalog/APIMAP.md)
- [Acquisizione](acquisition/APIMAP.md)
- [Sorgenti](sources/APIMAP.md)
- [Formati](formats/APIMAP.md)
- [Ispezione GGUF](tools/gguf/APIMAP.md)
