# Modelli / models

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [MODEL](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `catalog(query) -> ModelCatalog [proposta]`
- `resolve(reference) -> ModelArtifact [proposta]`
- `acquire(request) -> AcquisitionResult [proposta]`
- `verify(artifact) -> VerificationReport [proposta]`

Dipendenze: nessuna.

## Contratti delegati

- [Catalogo](catalog/APIMAP.md)
- [Acquisizione](acquisition/APIMAP.md)
- [Sorgenti](sources/APIMAP.md)
- [Formati](formats/APIMAP.md)
- [Ispezione GGUF](tools/gguf/APIMAP.md)
