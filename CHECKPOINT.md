# Ruth: checkpoint di progetto

Revisione documentale: 13 (indice main). Protocollo CONT-001 rev 1.
Fase corrente: **3 - migrazione incrementale per componente**.
Attivita' completata: `MIG-001 / PY-001`.
Stato: **prima migrazione tecnicamente integrata e accettata con collaudo personale**.

## Completato

- SETUP-001: continuita' GitHub fra ChatGPT e Codex collaudata nella fase iniziale.
- STRUCT-001: scaffold e mappe ricorsive pubblicati.
- PY-SCOPE-001: perimetro funzionale approvato dall'utente il 16 ottobre 2026.
- PY-REV-001: revisione tecnica dell'ispezione Python.
- PY-PROMOTE-001: PR #2 nel branch environment/python/main, revisione b80520ad903ff18f93398725ea95cd91f51da5d8.
- PY-PROMOTE-002: PR #3 in environment/main, merge fd61723874052053f38446ee82eca954870804e1.
- PY-PROMOTE-003: PR #8 in main, merge ce1b0bb47b4ae4ad0da40f3f83c964b66015680f.
- Ingresso Ruth `python3 main.py` integrato: verifica in sola lettura dell'interprete e della .venv con esito e motivi a terminale.
- PY-MANUAL-001: **complete**. L'utente ha comunicato esito conforme dell'avvio
  personale di Ruth e ha confermato mediante `git rev-parse HEAD` la revisione
  effettivamente provata: `ef4654af8ab106ef194af254d9f3884e5cabb6b0` (16 ottobre 2026).

## Verifiche e limiti

- GitHub Actions run 38885972247 e run 38886317198: success,
  test sintetici Python, API comuni, ingresso Ruth e coerenza delle mappe.
- Controllo del checkpoint su GitHub completato dopo l'integrazione.
- Collaudo manuale positivo riferito *esclusivamente* all'ispezione Python
  sull'ingresso Ruth alla revisione `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- Il controllo non certifica versioni dei pacchetti, GPU, backend nativi,
  benchmark o funzionalita' ancora pianificate.
- Nessun log, inventario reale o percorso di host e' pubblicato (DATA-001).

## Sincronizzazione locale (REMOTE-001)

- Copia utilizzata per il collaudo: HEAD verificato `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
  I commit di documentazione successivi non alterano il comportamento
  dell'ispezione; una nuova sincronizzazione con il main corrente resta
  da verificare separatamente.
- `local_sync: pending_no_access` per la copia di collaudo rispetto al
  nuovo commit documentale finale: fornire l'istruzione di aggiornamento
  fast-forward e ottenere riscontro, senza reinterpretare il test gia'
  accettato come non valido.
- Il checkout preesistente `environment/python/work/entrypoint`, osservato
  con modifiche locali, resta preservato e non sincronizzato automaticamente
  (`legacy_checkout: blocked_divergence`). Non scartare dati o modificarlo.

## Prossima azione

La migrazione PY-001 e' **chiusa**. Prima di migrare un nuovo componente,
presentare all'utente scopo, funzioni, ingressi, risultati, effetti,
limiti e verifiche del nuovo perimetro; attendere un nuovo semaforo
verde (MIG-FLOW-001). Pianificazione storica: dipendenze generiche
e diagnostica dell'ambiente; nessuna porzione successiva iniziata.
Kaggle resta inattivo.

## Decisioni vigenti

CONT-001, MIG-FLOW-001, MIG-FLOW-002, REMOTE-001, DATA-001,
PLAN-001 e SCOPE-001. SYNC-001 resta storica e superseded.

## DEP-001: seconda migrazione avviata (16 ottobre 2026)

- DEP-SCOPE-001 approvato dall'utente, compresa la CLI `main.py dependencies`,
  DEP-001 prima della migrazione hardware, diagnostica DIAG-001 separata.
- Codice candidato nel branch `environment/dependencies/work/DEP-001`, senza
  integrazione a `main` e senza verifica conclusiva: non dichiararlo finito.
- Stato: implementation complete (ambito approvato), verification not_run,
  integration not_merged, manual acceptance pending.
- Prossimo passo: test automatici, mappe, PR verso branch foglia e padri
  e ingresso reale nel main, quindi sync locale e collaudo personale.
- REMOTE-001: nessuna modifica al checkout locale; dati reali esclusi.
