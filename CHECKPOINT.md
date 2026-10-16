# Ruth: checkpoint di progetto

Revisione documentale: 15 (indice main). Protocollo CONT-001 rev 1.
Fase 3 - migrazione incrementale per componente.
Stato: **PY-001 e DEP-001 tecnicamente integrate, verificate e accettate
manualmente; orchestrazione MAIN-FLOW-001 approvata e in verifica tecnica**.

## Migrazioni completate

- PY-001: accettazione PY-MANUAL-001 positiva sul commit
  `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
  Codice read-only in main; PR #2, #3 e #8, test CI.
- DEP-001: DEP-SCOPE-001 approvato; promozioni PR #12, #13 e #15;
  CI GitHub 38895348468, 38895479549, 38895836703 e 38895924627
  success (14 Python, 12 dipendenze, 13 root/CLI, 237 mappe).
- DEP-MANUAL-001: **complete**. L'utente ha sincronizzato il clone
  pulito main mediante `git merge --ff-only origin/main`, confermato
  `git rev-parse HEAD` a `e197cc86829acf52e326a712331cc7ff6cfef6b5` e avviato personalmente
  `.venv/bin/python main.py dependencies pip`. Esito positivo:
  `present_unverified` per `pip` e riepilogo esplicito di
  compatibilita' non verificata (16 ottobre 2026).
- L'esito di un metadato presente non certifica compatibilita', sicurezza,
  installabilita', GPU o runtime. Nessun dato host o log in Git (DATA-001).

## Nuova decisione del flusso (MAIN-FLOW-001)

- L'utente ha chiesto esplicitamente il 16 ottobre 2026 che `main.py`
  esegua automaticamente **tutti i passaggi integrati** uno dopo l'altro;
  i sottocomandi restano opzionali per diagnosi.
- Prima sequenza candidata: isolamento .venv PY-001, poi ispezione metadati
  `pip` DEP-001. `pip` e' un controllo minimo iniziale; non e' un
  inventario dei requisiti complessivi di Ruth.
- Se Python non e' conforme, il controllo dipendenze viene saltato con
  motivazione e codice nonzero, senza interrogazioni ai pacchetti.
- Stato: codice candidato sul branch `runtime/work/MAIN-FLOW-001`;
  **test della nuova orchestrazione ancora da eseguire**, nessun nuovo
  merge o collaudo default effettuato. Non dichiarare verificata/integrata
  la modifica prima dell'esito CI e del merge.
- Prossima azione: test sintetici, mappe, PR verso main e verifica CI,
  aggiornamento di checkpoint e mappe, sincronizzazione locale finale.
  Presentare poi il comando `.venv/bin/python main.py` per il collaudo
  personale dell'avvio sequenziale prima di qualsiasi componente nuovo.

## Sincronizzazione locale

- La copia pulita di Ruth risulta sincronizzata al commit
  `e197cc86829acf52e326a712331cc7ff6cfef6b5` all'atto del collaudo DEP-001.
- Il checkout storico con modifiche locali resta separato e non va
  toccato (`legacy_checkout: blocked_divergence`).
- `local_sync: pending_final_merge` per la nuova orchestrazione.
  Seguire REMOTE-001: aggiornare il clone pulito solo dopo merge main,
  in fast-forward e preservando ogni modifica dell'utente.

## Decisioni correnti

CONT-001, MIG-FLOW-001/002, REMOTE-001, DATA-001, PLAN-001,
DEP-ORDER-001 e MAIN-FLOW-001. SYNC-001 resta superseded.
Kaggle inattivo, diagnostica avanzata e hardware non migrati.
