# Ruth: checkpoint di progetto

Revisione documentale: 19 (indice main), CONT-001 revisione 1.
Fase 3 - migrazione incrementale per componente.
Stato attuale: **DIAG-001 implementata, verificata e integrata in main;
accettazione personale DIAG-MANUAL-001 ancora in attesa**.

## Perimetri precedenti accettati

- PY-001: ispezione read-only Python/.venv integrata e collaudata
  positivamente dall'utente sul commit
  `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- DEP-001: presenza e confronto versioni esatte di pacchetti Python
  dichiarati, in sola lettura; collaudo utente positivo sul commit
  `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- MAIN-FLOW-001: sequenza automatica da `main.py`, collaudata
  positivamente sul commit
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
  Queste accettazioni restano valide.

## DIAG-001 integrata

- DIAG-SCOPE-001: perimetro approvato esplicitamente dall'utente
  il 16 ottobre 2026. Interpretazione read-only di report PY e DEP,
  categorie e suggerimenti portabili, senza intervenire sull'ambiente.
- API `environment.diagnostics.inspect(python_report, dependencies=None)`
  -> `DiagnosticReport` immutabile. Classifica configurazione .venv,
  pacchetto mancante, incompatibile o non ispezionabile, senza
  indovinare errori CUDA o altri problemi nativi.
- `main.py` senza argomenti ora esegue PY -> DEP -> DIAG. Se PY non
  e' conforme, salta DEP ma DIAG spiega comunque il report Python.
  I sottocomandi specifici restano disponibili; controllo pacchetto
  predefinito limitato a `pip` (non elenco completo delle dipendenze).
- Promozioni: PR #20 in environment/diagnostics/main
  (merge `ec7eba74bc71e62ea8562fa51d62295f5f3c01b8`);
  PR #21 in environment/main
  (merge `69ae0821cc0043bd76254bc46dff12b40d0d7b30`);
  PR #22 in main (merge `dfdb41e981f32ee233320f77998aadab55ce4d7b`).
- CI GitHub Actions: 38908192677, 38908269925 e 38908370200:
  **success**. 14 test Python, 12 dipendenze, 13 diagnostica,
  15 ingresso/root e mappe coerenti. Verifiche sintetiche.
- **DIAG-MANUAL-001 active**: l'utente non ha ancora confermato
  l'esecuzione personale del nuovo flusso DIAG sul commit finale.
  Nessuna successiva migrazione autorizzata senza questa verifica.

## Limiti e dati

- Nessun controllo hardware, CUDA, driver, log nativi, riparazione,
  installazione automatica, inferenza, benchmark o certificazione runtime.
- DATA-001: log, dati host, credenziali e versioni reali restano locali.
  Kaggle inattivo.
- REMOTE-001: checkout principale non modificato durante lo sviluppo.
  La copia pulita collaudata prima di DIAG deve essere sincronizzata
  dopo l'integrazione, in fast-forward e senza scartare lavoro.
  `local_sync: pending_no_access` per questa revisione.
  Il clone storico con modifiche locali rimane intatto
  (`legacy_checkout: blocked_divergence`).

## Prossima azione

Sincronizzare la copia pulita `ruth-main` solo dopo averne verificato
remote, branch `main` e assenza di modifiche locali da preservare.
Eseguire `.venv/bin/python main.py` e confermare SHA e risultato
dell'ordine PY -> DEP -> DIAG. Fino all'accettazione manuale,
la migrazione successiva e' bloccata (MIG-FLOW-002).
Dopo DIAG-MANUAL-001, presentare il prossimo perimetro e attendere
nuovo semaforo verde (MIG-FLOW-001).

## Decisioni vigenti

CONT-001, DATA-001, PLAN-001, SCOPE-001, MIG-FLOW-001/002,
REMOTE-001, MAIN-FLOW-001, DEP-ORDER-001 e DIAG-SCOPE-001.
SYNC-001 rimane superseded.
