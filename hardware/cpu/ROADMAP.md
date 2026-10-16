# CPU / hardware/cpu

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [CPU](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Inventario CPU portabile.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: not_run; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: CPU-001: architettura e numero logico tramite platform.machine/os.cpu_count, simulazione sintetica anche di valori mancanti; CI non ancora svolta. Non conta core fisici o GPU.

## Attivita locali

1. `CPU-001` [active] Implementato rilevamento CPU portabile, unknown/partial quando osservazione non disponibile; test e promozione da verificare.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
