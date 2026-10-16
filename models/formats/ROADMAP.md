# Formati / models/formats

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [FORMAT](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Contratto comune di lettura metadati e verifica.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 pending; no model payload or inference.

## Attivita locali

1. `FORMAT-001` [complete] Contratto formato minimo delegato al lettore GGUF header-only, verificato in CI e integrato in main via PR #44; formati completi futuri fuori perimetro.

## Componenti (ordine locale, ID stabili)

1. [GGUF: GGUF](gguf/ROADMAP.md) — active; complete; passed.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
