# Ruth: checkpoint di progetto

Revisione documentale: 22 (indice main), protocollo CONT-001 rev 1.
Fase 3: migrazione incrementale per componente.
Stato: **PY-001, DEP-001, DIAG-001 e HW-001/CPU-001 implementate,
verificate, integrate in main e accettate con collaudo personale**.

## Collaudi confermati

- PY-001: controllo Python/.venv accettato sulla revisione
  `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- DEP-001: dipendenze Python generiche, collaudo
  `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- MAIN-FLOW-001: avvio sequenziale di Ruth, collaudo
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
- DIAG-001: diagnostica read-only, collaudo
  `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`.
- HW-SCOPE-001 autorizzato il 16 ottobre 2026: rilevare CPU
  (architettura, processori logici) e RAM fisica totale, con valori
  unknown/partial dove necessario; GPU e driver in componenti separati.
- HW-001/CPU-001 integrati con PR #25 (hardware/cpu/main),
  PR #26 (hardware/main) e PR #27 (main;
  merge `35a6ac79a1b1287090c1a3d1dc9208d4129a15ec`).
  CI GitHub 38913317440, 38913438829, 38913533057: success;
  74 test sintetici e 249 mappe coerenti.
- **HW-MANUAL-001: complete**. L'utente ha sincronizzato la copia pulita
  di main tramite fast-forward, confermato HEAD
  `59b26fd6b427e6756ce0285fee60590cb077ebf1` ed eseguito personalmente `.venv/bin/python main.py`
  senza argomenti il 16 ottobre 2026. Sequenza PY -> DEP -> DIAG -> HW
  verificata, CPU e RAM riportate con stato `available`, riepilogo
  finale positivo. Non pubblicare caratteristiche o valori reali
  della macchina nel repository (DATA-001).

## Ambito verificato e limiti

- Ingresso Ruth: PY -> DEP -> DIAG -> HW. HW usa soltanto API
  standard library portabili e restituisce report transitori, non
  certificati di compatibilita' o prestazione.
- Il controllo predefinito delle dipendenze riguarda solo `pip`;
  `present_unverified` e `ok_unverified` non certificano
  integrita', sicurezza o compatibilita' runtime.
- Nessuna GPU NVIDIA/AMD/Intel o driver rilevati, nessun CUDA/ROCm/SYCL,
  inferenza, benchmark, installazione o modifica automatica dell'host.
- DATA-001: credenziali, output locali, percorsi personali,
  inventari reali hardware e dati host restano esclusi da Git.
  Kaggle inattivo.

## Sincronizzazione e sviluppo remoto (REMOTE-001)

- `local_sync: verified_on_tested_revision` per la copia pulita
  `ruth-main` sulla revisione `59b26fd6b427e6756ce0285fee60590cb077ebf1`.
- La chiusura documentale post-collaudo aggiunge nuovi commit;
  `local_sync: pending_docs_fast_forward` fino a nuova verifica.
  Il precedente collaudo resta valido: il codice applicativo
  non e' alterato dalla sola chiusura documentale.
- `legacy_checkout: blocked_divergence`: il vecchio checkout con
  modifiche locali preesistenti resta intatto. Non effettuare
  sync distruttive su quello storico.

## Prossimo passo

HW-MANUAL-001 e MIG-004 **complete**, nessun blocco tecnico/manuale
sulla porzione corrente. Non iniziare una migrazione GPU o un altro
componente senza descrivere scopo, ingressi, risultati, controlli,
effetti, limiti, ingresso sequenziale `main.py` e verifiche e ricevere
un **nuovo semaforo verde** dell'utente (MIG-FLOW-001).
Dopo ogni porzione integrata in main resta necessario il
collaudo utente sulla revisione provata (MIG-FLOW-002).

## Decisioni vigenti

CONT-001, PLAN-001, SCOPE-001, MIG-FLOW-001/002, REMOTE-001,
MAIN-FLOW-001, DATA-001, DEP-ORDER-001, DIAG-SCOPE-001,
HW-SCOPE-001. SYNC-001 e' storica e superseded.
