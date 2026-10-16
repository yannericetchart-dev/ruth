# Environment: indice di lavoro in main

Lavoro attivo: PY-001, prima porzione in
[PR #2](https://github.com/yannericetchart-dev/ruth/pull/2).
Il branch environment/main conserva lo scaffold. Il codice e i test della
porzione sono integrati in environment/python/main; stato aggiornato alla
revisione `b80520ad903ff18f93398725ea95cd91f51da5d8`.

PY-REV-001 completata: configurazioni ripetute malformate rifiutate e
catena reale delle API verificata. PY-PROMOTE-001 completata: PR #2 integrata
in environment/python/main.

Revisione funzionale PY-SCOPE-001 approvata dall'utente il 16 ottobre 2026: vedere
[descrizione delle funzioni](python/MIGRATION_SCOPE.md). Dopo il verde,
completare le promozioni a environment/main e main senza ulteriori conferme
intermedie; fermarsi per dubbi o decisioni da chiarire (MIG-FLOW-001).

MIG-FLOW-002 richiede esecuzione della funzione dal punto d'ingresso di Ruth
in main e messaggi a terminale. Dopo l'integrazione, il collaudo locale
personale dell'utente (PY-MANUAL-001) e la sua conferma positiva precedono
qualunque migrazione successiva. Il collegamento e il collaudo restano da fare.

Esaminare il checkpoint Python nella revisione di lavoro prima della
promozione. Le API indicate come proposte nelle mappe main restano tali
finche il codice non e' stato integrato lungo i branch padri.
