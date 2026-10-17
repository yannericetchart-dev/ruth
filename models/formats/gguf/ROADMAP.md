# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Lettura strutturale del formato, separata dagli strumenti CLI.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: partial.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference. GGUF-META-001: CI success runs 38940033886, 38940265778, 38940360999, 38940625555; PR #47 gguf, #48 formats, #49 models, #50 main (a630c83f6c62a229e18fe5e64f0863c19de16018). Catalog-only bounded KV reader, architecture/file_type/context_length optional; eight-step Ruth, synthetic tests and generated maps passed. GGUF-META-MANUAL-001 accepted on d318fd793da514f21ee2e8475304d1a6c883ee94 (catalog absent; no real GGUF tested); no tensors/download/inference. GGUF-INDEX-001: CI success runs 38945221302, 38945375637, 38945463381, 38945568401. PR #53 gguf, #54 formats, #55 Models, #56 main (1d51cfeed2bec83119ebf56ff0cb3840089484cf). Bounded tensor-index structural checks; unknown/quantized sizes unverified; no tensor payload. GGUF-INDEX-MANUAL-001 accepted personally on 09045243e944514d40b9e58c6159b013efa2a196 (catalog not configured; no real GGUF tested). GGUF-BLOCK-001 approved candidate; block synthetic CI not yet verified; previous nine-step scopes remain passed.

## Attivita locali

1. `GGUF-001` [complete] Lettore bounded 24-byte GGUF v2/v3: magic, versione, contatori e endian; validato con fixture sintetiche e integrato in main via PR #44.
2. `GGUF-MANUAL-001` [complete] User explicitly confirmed successful seven-step Ruth run on main ab81696dc93bc5d71a208cbcdc9a6f714b462fa9; GGUF catalog not_configured is valid; real GGUF file not exercised.
3. `GGUF-META-001` [complete] GGUF-META-SCOPE-001 bounded technical metadata for catalog GGUF: implemented, synthetic tests and CI passed, integrated in main via PR #50.
4. `GGUF-META-MANUAL-001` [complete] User explicitly accepted personal eight-step Ruth run on main d318fd793da514f21ee2e8475304d1a6c883ee94; GGUF and metadata not_configured valid, no real GGUF tested.
5. `GGUF-INDEX-001` [complete] GGUF-INDEX-SCOPE-001 descriptor-only checker implemented, synthetic CI verified and integrated in main PR #56.
6. `GGUF-INDEX-MANUAL-001` [complete] User explicitly accepted successful nine-step Ruth run on main 09045243e944514d40b9e58c6159b013efa2a196; INDEX not_configured is valid, no real model or weights inspected.
7. `GGUF-BLOCK-001` [active] Five GGML block extents Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 via descriptor-only reader, candidate CI/promotions pending.
8. `GGUF-BLOCK-MANUAL-001` [planned] Personal default ten-step Ruth run after main integration and explicit positive acceptance required.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
