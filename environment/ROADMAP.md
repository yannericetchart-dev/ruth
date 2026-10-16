# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Ispezionare Python e pianificare dipendenze senza installazioni implicite.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: stale; integrazione: partial.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DIAG-001: test sintetici della traduzione di report Python/dipendenze e dell'ordine PY->DEP->DIAG ancora da eseguire in CI; nessun controllo native/CUDA, nessuna compatibilita' certificata.

## Attivita locali

1. `ENV-001` [complete] Porzione PY-001 testata e promossa da environment/python/main a environment/main (PR #3) e main (PR #8).
2. `ENV-002` [complete] DEP-001 verificata e promossa in environment/main via PR #13 e in main via PR #15.
3. `ENV-003` [active] Integrare la diagnostica dei report PY e DEP nel flusso principale; test e promozioni ancora pendenti.

## Componenti (ordine locale, ID stabili)

1. [PY: Python](python/ROADMAP.md) — active; partial; passed.
2. [DEP: Dipendenze](dependencies/ROADMAP.md) — active; complete; passed.
3. [DIAG: Diagnostica](diagnostics/ROADMAP.md) — active; complete; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
