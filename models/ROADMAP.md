# Modelli / models

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [MODEL](NODE.yaml). Ripresa: [START_HERE](../START_HERE.md).

Catalogare, risolvere, acquisire e verificare artefatti modello.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: CAT-001: CI GitHub 38928909391, 38929028829, 38929118836 success (116 test sintetici inclusi 21 catalogo e 20 CLI, 264 mappe). PR #36 models/catalog/main, #37 models/main e #38 main. Manifest locale read-only, file solo dichiarati, niente download, lettura GGUF, inferenza o dati reali Git. CAT-MANUAL-001 confermato positivo sul commit 90e57f106fb664b0257775260044125ea85228e2. GGUF-001: CI success runs 38935705385, 38935861881, 38935943414, 38936032188; PR #41 gguf/main, #42 formats/main, #43 models/main, #44 main (15b106082c4dee37622c61a8afefd248a57464a0). Bounded 24-byte GGUF v2/v3 header, catalog-bound; synthetic tests and recursive maps passed. GGUF-MANUAL-001 accepted on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9 (catalog not configured; no real GGUF read); no model payload or inference. GGUF-META-001: CI success runs 38940033886, 38940265778, 38940360999, 38940625555; PR #47 gguf, #48 formats, #49 models, #50 main (a630c83f6c62a229e18fe5e64f0863c19de16018). Catalog-only bounded KV reader, architecture/file_type/context_length optional; eight-step Ruth, synthetic tests and generated maps passed. GGUF-META-MANUAL-001 pending; no tensors/download/inference.

## Attivita locali

1. `MODEL-001` [active] Catalogo locale CAT-001 verificato, integrato in main tramite PR #38 e accettato personalmente (CAT-MANUAL-001); altri componenti Models restano skeleton e richiedono nuova approvazione.
2. `MODEL-002` [complete] GGUF-001 header-only catalog reader CI-verified and integrated via PR #44; manual GGUF-MANUAL-001 passed on ab81696dc93bc5d71a208cbcdc9a6f714b462fa9. Other Models work still requires scope approval.
3. `MODEL-003` [active] GGUF-META-001 bounded catalog-only metadata reader CI verified and integrated via PR #50; manual GGUF-META-MANUAL-001 pending, no tensors/download/inference.

## Componenti (ordine locale, ID stabili)

1. [CAT: Catalogo](catalog/ROADMAP.md) — active; complete; passed.
2. [ACQ: Acquisizione](acquisition/ROADMAP.md) — inactive; skeleton; not_run.
3. [SOURCE: Sorgenti](sources/ROADMAP.md) — inactive; skeleton; not_run.
4. [FORMAT: Formati](formats/ROADMAP.md) — active; partial; passed.
5. [GGUFTOOL: Ispezione GGUF](tools/gguf/ROADMAP.md) — inactive; skeleton; not_run.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
