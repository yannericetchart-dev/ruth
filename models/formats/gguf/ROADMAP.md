# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Lettura strutturale del formato, separata dagli strumenti CLI.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference. GGUF-META-001: CI success runs 38940033886, 38940265778, 38940360999, 38940625555; PR #47 gguf, #48 formats, #49 models, #50 main (a630c83f6c62a229e18fe5e64f0863c19de16018). Catalog-only bounded KV reader, architecture/file_type/context_length optional; eight-step Ruth, synthetic tests and generated maps passed. GGUF-META-MANUAL-001 pending; no tensors/download/inference.

## Attivita locali

1. `GGUF-001` [complete] Lettore bounded 24-byte GGUF v2/v3: magic, versione, contatori e endian; validato con fixture sintetiche e integrato in main via PR #44.
2. `GGUF-MANUAL-001` [complete] User explicitly confirmed successful seven-step Ruth run on main ab81696dc93bc5d71a208cbcdc9a6f714b462fa9; GGUF catalog not_configured is valid; real GGUF file not exercised.
3. `GGUF-META-001` [complete] GGUF-META-SCOPE-001 bounded technical metadata for catalog GGUF: implemented, synthetic tests and CI passed, integrated in main via PR #50.
4. `GGUF-META-MANUAL-001` [active] Personal eight-step Ruth run and explicit positive acceptance required on synced main a630c83f6c62a229e18fe5e64f0863c19de16018; unconfigured catalog is valid.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
