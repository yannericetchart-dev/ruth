# GGUF / models/formats/gguf

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [GGUF](NODE.yaml). Ripresa: [START_HERE](../../../START_HERE.md).

Lettura strutturale del formato, separata dagli strumenti CLI.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: not_run; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

## Attivita locali

1. `GGUF-001` [active] Lettore bounded dei 24 byte iniziali per v2/v3, controllo magic, versioni, contatori, endian, errori/percorsi sicuri; candidato sul work branch, verifica e merge pendenti.
2. `GGUF-MANUAL-001` [planned] Dopo merge in main serve esecuzione personale di main.py senza argomenti e conferma settima fase (skip valido quando catalogo assente).

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
