# Modelli / models

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [MODEL](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Catalogare, risolvere, acquisire e verificare artefatti modello.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: CAT-001: CI GitHub 38928909391, 38929028829, 38929118836 success (116 test sintetici inclusi 21 catalogo e 20 CLI, 264 mappe). PR #36 models/catalog/main, #37 models/main e #38 main. Manifest locale read-only, file solo dichiarati, niente download, lettura GGUF, inferenza o dati reali Git. CAT-MANUAL-001 ancora da eseguire.

## Attivita locali

1. `MODEL-001` [active] Catalogo locale CAT-001 verificato e integrato in main tramite PR #38; altri componenti models restano skeleton; collaudo CAT-MANUAL-001 pendente.

## Componenti (ordine locale, ID stabili)

1. [CAT: Catalogo](catalog/ROADMAP.md) — active; complete; passed.
2. [ACQ: Acquisizione](acquisition/ROADMAP.md) — inactive; skeleton; not_run.
3. [SOURCE: Sorgenti](sources/ROADMAP.md) — inactive; skeleton; not_run.
4. [FORMAT: Formati](formats/ROADMAP.md) — inactive; skeleton; not_run.
5. [GGUFTOOL: Ispezione GGUF](tools/gguf/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
