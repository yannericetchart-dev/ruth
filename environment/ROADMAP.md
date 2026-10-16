# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Ispezionare Python e pianificare dipendenze senza installazioni implicite.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: not_merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: PY-001/PY-REV-001: isolamento Python, catena reale delle API e coerenza delle mappe su fixture sintetiche; nessuna verifica nativa.

## Attivita locali

1. `ENV-001` [active] Prima porzione Python integrata nel branch canonico del componente; da promuovere a environment/main. Dipendenze e diagnostica pianificate.

## Componenti (ordine locale, ID stabili)

1. [PY: Python](python/ROADMAP.md) — active; partial; passed.
2. [DEP: Dipendenze](dependencies/ROADMAP.md) — inactive; skeleton; not_run.
3. [DIAG: Diagnostica](diagnostics/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
