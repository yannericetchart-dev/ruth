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

## PY-PROMOTE-002: integrazione in environment/main (16 ottobre 2026)

- PR #3 integrata in environment/main al merge `fd61723874052053f38446ee82eca954870804e1`.
- Verifiche remote sulla revisione candidata `5fc816407f5c8873fd2ed39baa9d5293a9810c73`: GitHub Actions run `38885972247`, esito positivo (suite Python e test root, mappe coerenti, nessun file tracciato generato).
- API Python, catena environment/runtime e main.py integrate in environment/main; in main resta da promuovere. Nessun dato utente o log versionato.
- Prossimo passo: PR environment/main -> main, verifica sul commit effettivo e aggiornamento dei nodi/checkpoint fino alla radice. `local_sync: pending_final_merge`, collaudo PY-MANUAL-001 non eseguito.
