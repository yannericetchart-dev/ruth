# Hardware / hardware

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [HW](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Descrivere risorse senza confonderle con compatibilita di inferenza.

Sviluppo: **active**; codice: **complete**; maturita: experimental; verifica: not_run; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: HW-001/CPU-001: inventario read-only solo CPU e RAM con API stdlib, valori unknown dove non disponibili, test sintetici da eseguire in CI; GPU/vendor/backend esclusi. HW-MANUAL-001 ancora da svolgere dopo merge.

## Attivita locali

1. `HW-001` [active] Contratti HardwareReport/ResourceReport/DeviceInfo e inventario fisico RAM portabile, codice candidato e test sintetici/CI pendenti.
2. `HW-SCOPE-001` [complete] Funzioni, ingressi, risultati, effetti, limiti e verifiche con CPU/RAM prima di GPU approvati dall'utente il 16 ottobre 2026.
3. `HW-MANUAL-001` [planned] Dopo promozione main, utente avvia Ruth dal checkout pulito, verifica sequenza PY->DEP->DIAG->HW e SHA provato.

## Componenti (ordine locale, ID stabili)

1. [CPU: CPU](cpu/ROADMAP.md) — active; complete; not_run.
2. [GPU: GPU](gpu/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
