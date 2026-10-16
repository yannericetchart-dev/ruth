# Ruth: checkpoint di progetto

Revisione documentale: 20 (indice main), CONT-001 revisione 1.
Fase 3: migrazione incrementale per componente.
Stato corrente: **PY-001, DEP-001, DIAG-001 e flusso sequenziale
main.py integrati, verificati e accettati personalmente dall'utente**.

## Completato e accettato

- PY-001: ispezione read-only Python/.venv; PY-MANUAL-001 accettata
  sul commit `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- DEP-001: verifica in sola lettura dei metadati dei pacchetti espliciti
  e versioni esatte, collaudo DEP-MANUAL-001 accettato sul commit
  `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- MAIN-FLOW-001: ingresso Ruth senza argomenti esegue in sequenza le
  funzioni integrate, collaudo MAIN-MANUAL-001 accettato sul commit
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
- DIAG-SCOPE-001 approvato il 16 ottobre 2026: diagnostica read-only
  dei report gia' prodotti da Python e dipendenze, spiegazioni e
  suggerimenti senza modifiche host e senza controlli CUDA/GPU.
- DIAG-001 integrata attraverso PR #20 in environment/diagnostics/main
  (`ec7eba74bc71e62ea8562fa51d62295f5f3c01b8`), PR #21 in
  environment/main (`69ae0821cc0043bd76254bc46dff12b40d0d7b30`),
  PR #22 in main (`dfdb41e981f32ee233320f77998aadab55ce4d7b`).
- Test GitHub Actions 38908192677, 38908269925 e 38908370200:
  **success** (14 test PY, 12 DEP, 13 DIAG, 15 root/CLI, mappe).
- **DIAG-MANUAL-001 complete**: il 16 ottobre 2026 l'utente ha
  verificato dopo sincronizzazione non distruttiva che HEAD era
  `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb` e ha eseguito personalmente
  `.venv/bin/python main.py` senza argomenti. L'esito confermato
  mostra Python conforme, controllo minimo `pip` presente ma non
  certificato, avvio diagnostica, stato `ok_unverified` e riepilogo
  finale positivo, nell'ordine PY -> DEP -> DIAG.

## Limiti e dati

- Il controllo predefinito delle dipendenze riguarda solo `pip`,
  non l'inventario completo dei requisiti di Ruth.
- `present_unverified` e `ok_unverified` non certificano
  compatibilita' runtime, installabilita', sicurezza, GPU, CUDA o
  funzionamento dell'inferenza.
- Nessuna installazione, riparazione o modifica automatica dell'host.
- DATA-001: log, dati reali di host, credenziali, versioni locali e
  risultati benchmark non sono pubblicati su Git.
  Kaggle resta inattivo.

## Sincronizzazione REMOTE-001

- `local_sync: verified_on_tested_revision` per la copia pulita
  `ruth-main` sulla revisione `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`.
- Dopo la presente chiusura documentale, la nuova revisione GitHub
  deve essere applicata alla copia pulita con un fast-forward sicuro
  (`local_sync: pending_docs_fast_forward`); il precedente test
  resta valido, poiche' la chiusura modifica solo documentazione.
- Il clone storico con modifiche locali preesistenti resta preservato
  e non sincronizzato automaticamente
  (`legacy_checkout: blocked_divergence`).

## Prossima azione

DIAG-001 e MIG-003 **complete**; nessun blocco da collaudo residuo.
Prima di iniziare una nuova porzione, proporre scopo, ingressi,
risultati, controlli, effetti, limiti, ingresso `main.py` e test,
quindi attendere un **nuovo semaforo verde** (MIG-FLOW-001).
La migrazione hardware e gli altri domini restano non avviati.
Ogni nuova operazione approvata dovra' entrare nel flusso sequenziale
MAIN-FLOW-001 e richiedera' collaudo utente MIG-FLOW-002.

## Decisioni attive

CONT-001, MIG-FLOW-001/002, REMOTE-001, DATA-001, PLAN-001,
SCOPE-001, MAIN-FLOW-001, DEP-ORDER-001 e DIAG-SCOPE-001.
SYNC-001 resta storica e superseded.

## HW-001: perimetro approvato, candidato in verifica (16 ottobre 2026)

HW-SCOPE-001 approvato dall'utente: CPU (architettura e processori
logici), RAM fisica totale disponibile tramite API portabili, contratti
HardwareReport e ResourceReport. Provider GPU NVIDIA/AMD/Intel esclusi.
Code path `hardware/cpu`, `hardware/runtime` e quarta fase `main.py`.
Stato implementazione candidatura: complete; verification not_run;
integration not_merged; HW-MANUAL-001 planned.
Questa candidatura non misura memoria utilizzabile, benchmark, risorse GPU
o compatibilita' nativa. CI e promozioni ancora da compiere; local_sync
della nuova porzione pendente, clone storico intatto (REMOTE-001).
