# NVIDIA: checkpoint NVIDIA-001

## Stato del ref remoto (17 ottobre 2026)

Branch logico `hardware/gpu/nvidia/main`: **archived / ref remoto assente**.
Tip prima della rimozione: `60e137a422403efa7d8d4005240abf317ce87e24`.
Rimosso dall'esecutore in BRANCH-DELETE-002 (PR #109), verificato da B
tramite inventari completi heads/branches main-only. Conservazione: storia contenuta in main alla revisione
`e7efb79a86e28209528ebcf4d85ca41f526b9b39`.
NODE aggiornato ad archived, flag archived_branch e viste rigenerate.
Nomi e padri Git sotto restano provenienza e contratti logici, non ref presenti.
Codice, API e collaudi applicativi conservano il loro ambito precedente.
Per la coda documentale e lo stato finale leggere CHECKPOINT.md radice e
il checkpoint BRANCH-CLEANUP-001-FINAL sulla PR della registrazione.


GPU-001 NVIDIA **implementata, verificata, integrata e accettata
con collaudo personale positivo** il 16 ottobre 2026.

- API `hardware.gpu.nvidia.detect() -> GPUReport`, read-only,
  informazioni transitorie; nessun test driver/CUDA/backend incluso.
- Test sintetici in CI GitHub runs 38918374871, 38918650589,
  38918759116, 38918867451, 38919460341 tutti success;
  92 test complessivi e 258 mappe coerenti.
- PR #30, #31, #32 e #33 integrate fino a main;
  revisione verificata personalmente dall'utente `c344c71013cfa05da4f49f3204900f4bdc89b8ea`.
- **GPU-MANUAL-001 complete**: l'utente conferma quinto passaggio
  NVIDIA stato `available` e termine positivo del flusso Ruth.
  Specifiche hardware reali omesse (DATA-001).
- Prossima migrazione solo dopo perimetro nuovo e approvazione.
