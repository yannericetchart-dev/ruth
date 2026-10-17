# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Lettura strutturale del formato, separata dagli strumenti CLI.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference. GGUF-META-001: CI success runs 38940033886, 38940265778, 38940360999, 38940625555; PR #47 gguf, #48 formats, #49 models, #50 main (a630c83f6c62a229e18fe5e64f0863c19de16018). Catalog-only bounded KV reader, architecture/file_type/context_length optional; eight-step Ruth, synthetic tests and generated maps passed. GGUF-META-MANUAL-001 accepted on d318fd793da514f21ee2e8475304d1a6c883ee94 (catalog absent; no real GGUF tested); no tensors/download/inference. GGUF-INDEX-001: CI success runs 38945221302, 38945375637, 38945463381, 38945568401. PR #53 gguf, #54 formats, #55 Models, #56 main (1d51cfeed2bec83119ebf56ff0cb3840089484cf). Bounded tensor-index structural checks; unknown/quantized sizes unverified; no tensor payload. GGUF-INDEX-MANUAL-001 accepted personally on 09045243e944514d40b9e58c6159b013efa2a196 (catalog not configured; no real GGUF tested). GGUF-BLOCK-001: PR #59 gguf, #60 formats, #61 Models, #62 main (5f896f1bbacaa8c802f66f2a40ee7663cfa40604). CI runs 39022476943, 39022574327, 39022637716, 39022698747 passed (fixture sintetiche, entrypoint and maps); five GGML types 2/3/6/7/8 validated with block extents, no tensor payload; GGUF-BLOCK-MANUAL-001 personally accepted on 2f42a9d7a631e8779613784b58f8ce1b92afff9c (catalog not configured; no real GGUF tested). GGUF-KBLOCK-001: PR #65 GGUF, #66 Formats, #67 Models, #68 main (a803e15e872d71a96b7b2de37b13a8964d37d168). CI success 39025028855, 39025109274, 39025159357, 39025210189; synthetic K-block tests and generated maps passed. Six GGML type IDs 10-15 256-element extents. Legacy INDEX/BLOCKS unchanged; no tensor payload. GGUF-KBLOCK-MANUAL-001 personally accepted on 233efc358f56a66d2abf796fa8e20a476aa788cf (11 Ruth phases passed, 9 synthetic K-block tests OK; catalog not configured; no real GGUF verified). MIG-012 PR #71-#74 CI passed including synthetic fixtures, regression tests and recursive maps; main b057957b5d96b3c09ecde0b6c28ffc0e977c6942; personal test pending. GGUF-FIXTURE-MANUAL-001 personally accepted on a3ced17b0e4bccaf8fe8e10db9c1f5b5c508bda3 (12-step default Ruth, synthetic fixture passed with 3 files, 4 targeted tests OK; optional catalog absent; no real GGUF or tensor payload tested). MIG-013: PR #78-#81 CI passed including synthetic summary scenarios, CLI/regression tests and recursive maps; integrated main 6da03a348407c9caec355238c21c6a122a4b6298. Personal GGUF-STRUCT-SUMMARY-MANUAL-001 pending; no real GGUF or payload tested.

## Attivita locali

1. `GGUF-001` [complete] Lettore bounded 24-byte GGUF v2/v3: magic, versione, contatori e endian; validato con fixture sintetiche e integrato in main via PR #44.
2. `GGUF-MANUAL-001` [complete] User explicitly confirmed successful seven-step Ruth run on main ab81696dc93bc5d71a208cbcdc9a6f714b462fa9; GGUF catalog not_configured is valid; real GGUF file not exercised.
3. `GGUF-META-001` [complete] GGUF-META-SCOPE-001 bounded technical metadata for catalog GGUF: implemented, synthetic tests and CI passed, integrated in main via PR #50.
4. `GGUF-META-MANUAL-001` [complete] User explicitly accepted personal eight-step Ruth run on main d318fd793da514f21ee2e8475304d1a6c883ee94; GGUF and metadata not_configured valid, no real GGUF tested.
5. `GGUF-INDEX-001` [complete] GGUF-INDEX-SCOPE-001 descriptor-only checker implemented, synthetic CI verified and integrated in main PR #56.
6. `GGUF-INDEX-MANUAL-001` [complete] User explicitly accepted successful nine-step Ruth run on main 09045243e944514d40b9e58c6159b013efa2a196; INDEX not_configured is valid, no real model or weights inspected.
7. `GGUF-BLOCK-001` [complete] GGUF-BLOCK-SCOPE-001 five types Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 implemented, synthetic CI passed and integrated in main PR #62.
8. `GGUF-BLOCK-MANUAL-001` [complete] User explicitly accepted successful ten-stage Ruth main run on 2f42a9d7a631e8779613784b58f8ce1b92afff9c; BLOCKS not_configured valid, no real GGUF file or weight inspected.
9. `GGUF-KBLOCK-001` [complete] GGUF-KBLOCK-SCOPE-001 six K-quant types Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K implemented, synthetic CI passed and integrated in main PR #68.
10. `GGUF-KBLOCK-MANUAL-001` [complete] User supplied positive 11-stage Ruth output on main 233efc358f56a66d2abf796fa8e20a476aa788cf and local 9-test K-block fixture suite OK, deemed acceptance under MIG-FLOW-002 revision 2; CAT not configured, no real GGUF checked.
11. `GGUF-FIXTURE-001` [complete] MIG-012: GGUF-FIXTURE-001 implemented, CI passed and integrated via PR #71-#74; personal 12-stage Ruth plus 4 synthetic tests OK on a3ced17b0e4bccaf8fe8e10db9c1f5b5c508bda3, accepted under MIG-FLOW-002-R2. No real GGUF tested; next scope requires approval.
12. `GGUF-FIXTURE-MANUAL-001` [complete] Personal 12-step Ruth run on verified main a3ced17b0e4bccaf8fe8e10db9c1f5b5c508bda3 and targeted test_pipeline suite (4 tests, OK), accepted on 17 October 2026 without second approval; synthetic-only fixture, no real GGUF.
13. `GGUF-STRUCT-SUMMARY-001` [complete] MIG-013: GGUF structural summary and Ruth SUMMARY stage; verified by synthetic CI and integrated via PR #78-#81 on main 6da03a348407c9caec355238c21c6a122a4b6298. Personal GGUF-STRUCT-SUMMARY-MANUAL-001 pending; no real model/payload tested.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
