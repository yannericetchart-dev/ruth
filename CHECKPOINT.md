# Ruth: checkpoint di progetto

Revisione documentale: 23 (indice main); CONT-001 revisione 1.
Fase 3: migrazione incrementale per componente.
Stato: **GPU-001 (NVIDIA) implementata, verificata e integrata in main;
GPU-MANUAL-001 ancora in attesa di collaudo personale**.

## Migrazioni precedenti confermate

- PY-001: ispezione Python/.venv, accettata dall'utente sul commit
  `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- DEP-001: metadati e versioni esatte Python, accettata su
  `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- DIAG-001: diagnosi Python/dipendenze, accettata su
  `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`.
- HW-001/CPU-001: architettura CPU, processori logici e RAM totale,
  accettata dall'utente su
  `59b26fd6b427e6756ce0285fee60590cb077ebf1`.
- MAIN-FLOW-001: ingresso ufficiale senza argomenti, automatico e
  sequenziale, originariamente collaudato su
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
  Tutte le accettazioni precedenti restano valide.

## GPU-001: integrazione tecnica

- GPU-SCOPE-001 approvato dall'utente il 16 ottobre 2026: solo NVIDIA,
  tramite `nvidia-smi` a parametri fissi, senza shell e con timeout;
  dati GPU: modello, VRAM totale e numero di dispositivi rilevati.
  Stati distinti: available, partial, none_reported, unavailable,
  timeout, inspection_failed, unknown. AMD/Intel non attivate.
- Contratti `GPUReport` e `GPUDevice` immutabili e transitori;
  `hardware.gpu.nvidia.detect()` e `hardware.gpu.detect()`.
  Nessun seriale, UUID, identificativo persistente o stdout/stderr
  imprevisto conservato. Nessun driver o pacchetto installato.
- `main.py` ora esegue **PY -> DEP -> DIAG -> HW CPU/RAM -> NVIDIA**.
  GPU-001 e' indipendente dai controlli .venv; errore o assenza
  utility NVIDIA non causano un fallimento automatico delle altre fasi.
  Il sottocomando `dependencies` resta mirato.
- Promozioni: PR #30 in hardware/gpu/nvidia/main
  (merge `21c17032af358120b7cf0d56246a9c135f2833f7`),
  PR #31 in hardware/gpu/main
  (merge `0010c43b75c1f28c18b6c83dba0c913f384a40c2`),
  PR #32 in hardware/main
  (merge `a7c5e63ea460c00d9ff5c73913af3835c182b2e2`),
  PR #33 in main
  (merge `54a02cf37262a78f5a1ed0aa84cd6d210060a14c`).
- CI GitHub runs 38918374871, 38918650589, 38918759116,
  38918867451 **success**; 92 test sintetici (14 PY, 12 DEP,
  13 DIAG, 8 CPU, 12 HW/RAM, 16 NVIDIA, 17 CLI/root) e
  258 mappe coerenti. Non sono test GPU reali.
- Stato: implementation complete, verification passed (fixture
  sintetiche), integration merged; **GPU-MANUAL-001 active**.
  Nessun collaudo NVIDIA personale ancora confermato e nessun
  componente successivo autorizzato.

## Limiti e dati

- NVIDIA non disponibile, non riportata o ispezione fallita
  **non dimostrano** che una GPU NVIDIA sia fisicamente assente.
- Nessuna prova CUDA, driver, GPU AMD/Intel, VRAM libera, benchmark,
  backend llama.cpp, inferenza, compatibilita' runtime o prestazioni.
  `pip` resta l'unico controllo di dipendenza predefinito e
  `present_unverified` non certifica compatibilita'.
- DATA-001: inventari veri, log, benchmark, credenziali, percorsi
  utente e specifiche macchina rimangono locali ed esclusi da Git.
  Kaggle inattivo.

## Sincronizzazione e collaudo (REMOTE-001, MIG-FLOW-002)

- `local_sync: pending_no_access` per il clone pulito dell'utente
  dopo l'integrazione GPU; l'ultima sincronizzazione personalmente
  confermata riguardava la prova HW-001 precedente.
- Il checkout storico preesistente con modifiche rimane intatto:
  `legacy_checkout: blocked_divergence`.
- Dopo questa chiusura documentale, su `ruth-main` verificare che
  remote punti al repository canonico, branch `main` e stato pulito;
  applicare soltanto `git fetch origin`, `git merge --ff-only
  origin/main`, poi `git rev-parse HEAD`.
- Eseguire personalmente `.venv/bin/python main.py` e confermare
  commit esatto, ordine dei cinque passaggi, stato generico della
  quinta fase e messaggio finale. Nessun bisogno di trasferire al
  repository valori realmente osservati o log host.
- Accettazione **GPU-MANUAL-001 pendente**, impedisce qualsiasi
  migrazione successiva (MIG-FLOW-002). Dopo esito positivo, nuova
  porzione solo previa approvazione delle funzioni (MIG-FLOW-001).

## Decisioni vigenti

CONT-001, DATA-001, MIG-FLOW-001/002, REMOTE-001, MAIN-FLOW-001,
SCOPE-001, PLAN-001, HW-SCOPE-001, GPU-SCOPE-001.
SYNC-001 rimane storica/superseded.
