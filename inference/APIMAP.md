# Inferenza / inference

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [INF](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

## Contratto dichiarato

- `assess(request) -> StackAssessment [proposta]`
- `open(request) -> InferenceSession [proposta]`
- `session.describe/count_tokens/generate/stream/cancel/close [proposta]`

Dipendenze: [models](../models/APIMAP.md), [hardware](../hardware/APIMAP.md).

## Contratti delegati

- [Backend](backends/APIMAP.md)
- [Motori](engines/APIMAP.md)
