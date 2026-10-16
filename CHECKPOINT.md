# Ruth: checkpoint di progetto

Revisione documentale: 14 (indice main). Protocollo CONT-001 rev 1.
Fase: **3 - migrazione incrementale per componente**.
Porzione corrente: **MIG-002 / DEP-001**.
Stato: **integrazione tecnica completata; collaudo personale DEP-MANUAL-001 in attesa**.

## Passaggi completati

- SETUP-001 e STRUCT-001: continuita' GitHub tra client e scaffold del progetto verificati in precedenza.
- PY-001: implementata, verificata e integrata in main, accettazione
  manuale PY-MANUAL-001 **complete** sul commit
  `ef4654af8ab106ef194af254d9f3884e5cabb6b0` (16 ottobre 2026).
  PR #2 (foglia Python), PR #3 (environment) e PR #8 (main).
- DEP-SCOPE-001: perimetro read-only e CLI dipendenze approvato
  dall'utente il 16 ottobre 2026 (MIG-FLOW-001).
- DEP-PROMOTE-001: PR #12 in environment/dependencies/main,
  merge `4af2050e5fc59477f9ab854173bf915b7933d186`.
- DEP-PROMOTE-002: PR #13 in environment/main,
  merge `c89c02ac0afa828b2c53ebd922a04331069ba527`.
- DEP-PROMOTE-003: PR #15 in main,
  merge `a542f9ee9f6d09a0496bf756615d2295a7f59ce0`.
- Codice ufficiale in main: `environment.dependencies.plan`,
  `environment.plan`, report immutabili, `main.py dependencies`
  con messaggi e codici di uscita, oltre a `main.py` senza argomenti
  per il controllo Python esistente.

## Verifica e limiti DEP-001

- GitHub Actions run 38895348468, 38895479549, 38895836703 e
  38895924627: **success**; 14 test isolamento Python, 12 test
  dipendenze, 13 test root/CLI e 237 mappe coerenti sulla candidatura.
- Limiti: metadati Python, nomi e versioni esatte soltanto; `present_unverified`
  non e' prova di compatibilita' runtime, integrita', sicurezza o
  installabilita'. Nessun pip automatico, lock CUDA, test GPU o inferenza.
- DIAG-001 (diagnostica avanzata) resta pianificata e non autorizzata,
  l'hardware e' successivo; Kaggle inattivo.
- Nessun log, misura o dato reale di host pubblicato (DATA-001).

## Prossima azione: sincronizzazione e collaudo DEP-MANUAL-001

- `local_sync: pending_no_access` per il clone pulito utilizzato
  dall'utente: verificarne branch e modifiche, aggiornarlo in fast-forward
  al main pubblicato con questo checkpoint e confermare lo SHA.
- Il vecchio checkout con modifiche locali sul branch
  `environment/python/work/entrypoint` resta intatto e non sincronizzato
  automaticamente (`legacy_checkout: blocked_divergence`).
- Dalla copia pulita aggiornata alla revisione finale, eseguire di persona
  `.venv/bin/python main.py dependencies pip`. Attesi avvio controllo
  dipendenze, stato `present_unverified` per pip e riepilogo non certificante
  la compatibilita'. Provare anche, se desiderato, un nome certamente assente.
- Attendere esplicita conferma dell'utente e SHA della revisione provata;
  DEP-MANUAL-001 resta **active**. Non iniziare un nuovo componente prima
  del suo collaudo positivo (MIG-FLOW-002).

## Decisioni vigenti

CONT-001, DATA-001, PLAN-001, SCOPE-001, MIG-FLOW-001,
MIG-FLOW-002, REMOTE-001 e DEP-ORDER-001. SYNC-001 e' storica
e sostituita.
