# Python: indice di lavoro in main

PY-001 e' integrata in environment/python/main tramite
[PR #2](https://github.com/yannericetchart-dev/ruth/pull/2), merge
`8c76080df5ba34e354f484c92adf7e132b594c2d`. Stato del componente aggiornato
al commit `b80520ad903ff18f93398725ea95cd91f51da5d8`.

La sola ispezione Python e' implementata, testata e revisionata (PY-REV-001)
in quella revisione. PY-PROMOTE-001 completata; il codice non e' ancora
integrato in environment/main.

Revisione funzionale [PY-SCOPE-001](MIGRATION_SCOPE.md): semaforo verde
concesso dall'utente il 16 ottobre 2026. Dopo il verde, completare autonomamente le promozioni
a environment/main e main secondo MIG-FLOW-001, senza conferme a ogni merge.

Prima di passare alla migrazione successiva, PY-MANUAL-001 richiede che
l'utente avvii Ruth localmente dal punto d'ingresso di main, osservi i
messaggi della funzione e confermi l'esito positivo (MIG-FLOW-002).
Collegamento al punto d'ingresso e collaudo non ancora realizzati.
Questo branch main contiene lo scaffold, senza il codice della porzione.
Vedere il
[checkpoint della porzione](https://github.com/yannericetchart-dev/ruth/blob/b80520ad903ff18f93398725ea95cd91f51da5d8/environment/python/CHECKPOINT.md).

## PY-PROMOTE-002: integrazione in environment/main (16 ottobre 2026)

- PR #3 integrata in environment/main al merge `fd61723874052053f38446ee82eca954870804e1`.
- Verifiche remote sulla revisione candidata `5fc816407f5c8873fd2ed39baa9d5293a9810c73`: GitHub Actions run `38885972247`, esito positivo (suite Python e test root, mappe coerenti, nessun file tracciato generato).
- API Python, catena environment/runtime e main.py integrate in environment/main; in main resta da promuovere. Nessun dato utente o log versionato.
- Prossimo passo: PR environment/main -> main, verifica sul commit effettivo e aggiornamento dei nodi/checkpoint fino alla radice. `local_sync: pending_final_merge`, collaudo PY-MANUAL-001 non eseguito.
