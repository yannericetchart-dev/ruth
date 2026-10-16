# Python: checkpoint PY-001

PY-001 implementata, revisionata, verificata e integrata tecnicamente
in main; contratto sperimentale in sola lettura.

- PY-REV-001: correzione configurazione malformata ripetuta e verifica API.
- PY-PROMOTE-001: PR #2 in environment/python/main, b80520ad903ff18f93398725ea95cd91f51da5d8.
- PY-PROMOTE-002: PR #3 in environment/main, fd61723874052053f38446ee82eca954870804e1.
- PY-PROMOTE-003: PR #8 in main, ce1b0bb47b4ae4ad0da40f3f83c964b66015680f.
- PY-SCOPE-001: approvato dall'utente il 16 ottobre 2026.
- Verifiche remote: run 38885972247 e 38886317198, entrambe success.
  Fixture virtualenv sintetiche, tre API pubbliche, ingresso Ruth, mappe.
- Effetti: sola lettura, nessuna installazione o raccolta telemetria.

PY-MANUAL-001: `pending`. Dopo sincronizzazione sicura di main, l'utente
deve eseguire personalmente Ruth tramite `python3 main.py`, confermare
la revisione provata e l'esito positivo. Checkout locale precedente su
branch di lavoro con modifiche, `local_sync: blocked_divergence`;
nessun dato o lavoro locale viene scartato. Vietata migrazione successiva
prima della conferma positiva (MIG-FLOW-002).
