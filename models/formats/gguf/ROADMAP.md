# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Lettura strutturale del formato, separata dagli strumenti CLI.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference.

## Attivita locali

1. `GGUF-001` [complete] Lettore bounded 24-byte GGUF v2/v3: magic, versione, contatori e endian; validato con fixture sintetiche e integrato in main via PR #44.
2. `GGUF-MANUAL-001` [complete] User explicitly confirmed successful seven-step Ruth run on main ab81696dc93bc5d71a208cbcdc9a6f714b462fa9; GGUF catalog not_configured is valid; real GGUF file not exercised.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
