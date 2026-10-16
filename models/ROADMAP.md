# Modelli / models

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [MODEL](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Catalogare, risolvere, acquisire e verificare artefatti modello.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: stale; integrazione: partial.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: CAT-001: manifest JSON opzionale in .local/models/catalog.json, file dichiarati sotto .local/models senza symlink o traversal, report read-only, niente discovery/download/GGUF o dati reali su Git. Test sintetici e promozioni ancora da eseguire.

## Attivita locali

1. `MODEL-001` [active] Prima porzione del dominio: catalogo locale CAT-001 approvato e candidato; acquisizione, GGUF e sorgenti restano skeleton.

## Componenti (ordine locale, ID stabili)

1. [CAT: Catalogo](catalog/ROADMAP.md) — active; complete; not_run.
2. [ACQ: Acquisizione](acquisition/ROADMAP.md) — inactive; skeleton; not_run.
3. [SOURCE: Sorgenti](sources/ROADMAP.md) — inactive; skeleton; not_run.
4. [FORMAT: Formati](formats/ROADMAP.md) — inactive; skeleton; not_run.
5. [GGUFTOOL: Ispezione GGUF](tools/gguf/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
