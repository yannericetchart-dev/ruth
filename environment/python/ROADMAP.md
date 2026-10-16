# Python / environment/python

Generato da NODE.yaml con `python3 tools/maps.py --write`; modificare i metadati.

Nodo responsabile: [PY](NODE.yaml). Ripresa: [START_HERE](../../START_HERE.md).

Verificare interprete e isolamento del virtualenv.

Sviluppo: **active**; codice: **partial**; maturita: experimental; verifica: passed; integrazione: merged.

Flag: in_scope. Blocchi: nessuno registrato.

Ambito verifica: PY-001: GitHub Actions run 38885972247 e run 38886317198 conclusi con successo (test Python su fixture sintetiche, API pubbliche, test ingresso Ruth, test mappe e tools/maps.py --check). Non verifica GPU, pacchetti nativi o hardware; collaudo personale PY-MANUAL-001 pendente.

## Attivita locali

1. `PY-001` [complete] Ispezione Python e ingresso Ruth implementati, revisionati, verificati e integrati in main tramite PR #8.
2. `PY-REV-001` [complete] Revisione tecnica completata con correzione del caso malformato ripetuto e verifica delle API reali.
3. `PY-PROMOTE-001` [complete] Prima promozione nella foglia environment/python/main tramite PR #2, preservata nella storia.
4. `PY-SCOPE-001` [complete] Semaforo verde esplicito dell'utente del 16 ottobre 2026, incluso ingresso Ruth.
5. `PY-MANUAL-001` [active] In attesa di sincronizzazione locale sicura e collaudo personale di python3 main.py sulla revisione main; conferma esplicita necessaria prima di migrazione successiva.
6. `PY-PROMOTE-002` [complete] Promozione in environment/main completata con PR #3, merge fd61723874052053f38446ee82eca954870804e1.
7. `PY-PROMOTE-003` [complete] Promozione in main completata con PR #8, merge ce1b0bb47b4ae4ad0da40f3f83c964b66015680f.

Propagare impatti e blocchi al [padre](../ROADMAP.md) e proseguire fino alla radice.
