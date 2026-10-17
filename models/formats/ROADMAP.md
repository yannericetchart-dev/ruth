# Formati / models/formats

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [FORMAT](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Contratto comune di lettura metadati e verifica.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference. GGUF-META-001: CI success runs 38940033886, 38940265778, 38940360999, 38940625555; PR #47 gguf, #48 formats, #49 models, #50 main (a630c83f6c62a229e18fe5e64f0863c19de16018). Catalog-only bounded KV reader, architecture/file_type/context_length optional; eight-step Ruth, synthetic tests and generated maps passed. GGUF-META-MANUAL-001 accepted on d318fd793da514f21ee2e8475304d1a6c883ee94 (catalog absent; no real GGUF tested); no tensors/download/inference. GGUF-INDEX-001: CI success runs 38945221302, 38945375637, 38945463381, 38945568401. PR #53 gguf, #54 formats, #55 Models, #56 main (1d51cfeed2bec83119ebf56ff0cb3840089484cf). Bounded tensor-index structural checks; unknown/quantized sizes unverified; no tensor payload. GGUF-INDEX-MANUAL-001 accepted personally on 09045243e944514d40b9e58c6159b013efa2a196 (catalog not configured; no real GGUF tested). GGUF-BLOCK-001: PR #59 gguf, #60 formats, #61 Models, #62 main (5f896f1bbacaa8c802f66f2a40ee7663cfa40604). CI runs 39022476943, 39022574327, 39022637716, 39022698747 passed (fixture sintetiche, entrypoint and maps); five GGML types 2/3/6/7/8 validated with block extents, no tensor payload; GGUF-BLOCK-MANUAL-001 pending.

## Attivita locali

1. `FORMAT-001` [complete] Contratto formato minimo delegato al lettore GGUF header-only, verificato in CI e integrato in main via PR #44; formati completi futuri fuori perimetro.
2. `FORMAT-META-001` [complete] GGUF-META-001 delegated bounded metadata inspection CI verified and integrated via PR #50; remaining full format capabilities excluded.
3. `FORMAT-INDEX-001` [complete] GGUF-INDEX-001 delegated bounded tensor-index reader CI verified and merged to main via PR #56; full tensor decoding excluded.
4. `FORMAT-BLOCK-001` [complete] GGUF-BLOCK-001 five-type descriptor block layout verification passed CI, integrated via PR #62; no other quantized type support.

## Componenti (ordine locale, ID stabili)

1. [GGUF: GGUF](gguf/ROADMAP.md) — active; complete; passed.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
