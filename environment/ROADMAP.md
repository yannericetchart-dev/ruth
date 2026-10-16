# Ambiente / environment

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [ENV](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Ispezionare Python e pianificare dipendenze senza installazioni implicite.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: partial.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: DEP-001: GitHub Actions 38895348468 (14 Python, 12 dipendenze, 13 root/CLI, 237 mappe), e 38895479549 (promozione padre): success su fixture sintetiche; nessuna verifica nativa, sicurezza o installabilita' certificata. DEP-MANUAL-001 ancora da svolgere dopo main.

## Attivita locali

1. `ENV-001` [complete] Porzione PY-001 testata e promossa da environment/python/main a environment/main (PR #3) e main (PR #8).
2. `ENV-002` [active] DEP-001 integrata in environment/main con PR #13 e verificata; promozione a main in corso.

## Componenti (ordine locale, ID stabili)

1. [PY: Python](python/ROADMAP.md) — active; partial; passed.
2. [DEP: Dipendenze](dependencies/ROADMAP.md) — active; complete; passed.
3. [DIAG: Diagnostica](diagnostics/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
