# Ruth: checkpoint di progetto

Revisione documentale: 21 (indice main), protocollo CONT-001 rev 1.
Fase 3: migrazione incrementale per componente.
Stato attuale: **HW-001/CPU-001 tecnicamente integrate e verificate;
collaudo personale HW-MANUAL-001 ancora da svolgere**.

## Componenti gia' accettati

- PY-001: ambiente Python/.venv in sola lettura, collaudo personale
  positivo sulla revisione `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- DEP-001: metadati di pacchetti Python dichiarati e confronto delle
  sole versioni esatte; collaudo personale positivo su
  `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- MAIN-FLOW-001: ingresso automatico sequenziale, accettato sul commit
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
- DIAG-001: diagnostica read-only dei report Python/dipendenze,
  collaudo personale positivo sul commit
  `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`.
  Le precedenti accettazioni rimangono valide.

## HW-001: integrazione completata

- HW-SCOPE-001 approvato esplicitamente dall'utente il 16 ottobre 2026:
  CPU (architettura, numero logico) e RAM fisica totale, con valori
  `unknown` dove non disponibili. GPU vendor rinviati a un perimetro
  separato. Contratti frozen `DeviceInfo`, `HardwareReport`,
  `ResourceReport`.
- Provider CPU stdlib: `platform.machine()` e `os.cpu_count()`;
  RAM totale (byte) tramite `os.sysconf()` quando supportato.
  Non e' memoria libera, utilizzabile dal processo o quota cgroup.
- Punto d'ingresso `main.py` senza argomenti ora esegue
  **PY -> DEP -> DIAG -> HW**. HW e' indipendente dalla conformita'
  Python, pertanto viene presentato anche se PY/DEP falliscono.
  Il controllo di pacchetti predefinito rimane limitato a `pip`
  e non e' la lista completa delle dipendenze di Ruth.
- Promozioni: PR #25 in hardware/cpu/main
  (merge `bf8c72e4566c22b768b36b770c66bbd0f770dbaa`);
  PR #26 in hardware/main
  (merge `d2ea16b14b89cefacea574f1e209fecec29f7f01`);
  PR #27 in main
  (merge `35a6ac79a1b1287090c1a3d1dc9208d4129a15ec`).
- CI GitHub Actions runs 38913317440, 38913438829, 38913533057:
  **success**, 74 test sintetici (14 PY, 12 DEP, 13 DIAG,
  8 CPU, 12 HW/RAM, 15 root/CLI), 249 mappe coerenti.
- Stato: implementation complete; verification passed (fixture
  sintetiche); integration merged. **HW-MANUAL-001 active**:
  l'utente non ha ancora collaudato la funzione nuova su main.

## Limiti e dati

- Nessuna GPU NVIDIA/AMD/Intel rilevata da HW-SCOPE-001; la loro presenza
  o compatibilita' CUDA/ROCm/SYCL, il backend, le prestazioni e
  l'inferenza non sono certificati.
- Nessuna installazione, riparazione, benchmark o modifica dell'host.
  I dati hardware osservati sono solo transitori e locali.
- DATA-001: non pubblicare inventari, log, dettagli reali macchina,
  dati privati, credenziali o versioni locali. Kaggle inattivo.

## Sincronizzazione e accettazione

- REMOTE-001: sviluppo e promozioni effettuati esclusivamente su GitHub.
  `local_sync: pending_no_access` per la copia pulita `ruth-main`
  rispetto alla revisione hardware definitiva.
- Il vecchio checkout con modifiche Codex rimane intatto
  (`legacy_checkout: blocked_divergence`). Non forzare sincronizzazioni.
- Dopo merge documentale, chiedere verifica `git status --short --branch`,
  branch `main`, fast-forward `git fetch origin && git merge --ff-only
  origin/main`, SHA con `git rev-parse HEAD`, poi avvio personale
  `.venv/bin/python main.py`.
- Atteso quarto passaggio `Ruth: avvio inventario hardware (CPU e RAM).`,
  linee di riepilogo CPU e RAM e stato `available`, `partial`
  o `unknown`; successo dei passaggi PY/DEP/DIAG precedenti deve
  continuare ad apparire.
- HW-MANUAL-001 resta da confermare esplicitamente con SHA e risultato
  della sequenza, secondo MIG-FLOW-002. **Non iniziare GPU-001 o un'altra
  migrazione** finche' HW-MANUAL-001 non sia accettata.

## Decisioni vigenti

CONT-001, PLAN-001, MIG-FLOW-001/002, REMOTE-001, DATA-001,
SCOPE-001, MAIN-FLOW-001, DEP-ORDER-001, DIAG-SCOPE-001,
HW-SCOPE-001. SYNC-001 e' storica/superseded.
