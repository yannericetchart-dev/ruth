# Python: indice di lavoro in main

PY-001 e' integrata in environment/python/main tramite
[PR #2](https://github.com/yannericetchart-dev/ruth/pull/2), merge
`8c76080df5ba34e354f484c92adf7e132b594c2d`. Stato del componente aggiornato
al commit `b80520ad903ff18f93398725ea95cd91f51da5d8`.

La sola ispezione Python e' implementata, testata e revisionata (PY-REV-001)
in quella revisione. PY-PROMOTE-001 completata; il codice non e' ancora
integrato in environment/main.

Revisione funzionale [PY-SCOPE-001](MIGRATION_SCOPE.md): semaforo verde concesso dall'utente il 16 ottobre 2026. Dopo il verde, completare autonomamente le promozioni
a environment/main e main secondo MIG-FLOW-001, senza conferme a ogni merge.

Prima di passare alla migrazione successiva, PY-MANUAL-001 richiede che
l'utente avvii Ruth localmente dal punto d'ingresso di main, osservi i
messaggi della funzione e confermi l'esito positivo (MIG-FLOW-002).
Collegamento al punto d'ingresso implementato nel branch di lavoro `environment/work/PY-001-entrypoint`, non ancora verificato/integrato; collaudo manuale non realizzato.
Questo branch main contiene lo scaffold, senza il codice della porzione.
Vedere il
[checkpoint della porzione](https://github.com/yannericetchart-dev/ruth/blob/b80520ad903ff18f93398725ea95cd91f51da5d8/environment/python/CHECKPOINT.md).
