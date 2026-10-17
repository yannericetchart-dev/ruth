# CPU: checkpoint CPU-001

## Stato del ref remoto (17 ottobre 2026)

Branch logico `hardware/cpu/main`: **archived / ref remoto assente**.
Tip prima della rimozione: `ae492a1be3b12c0ffd6f9987d0d959aceb4a9ad4`.
Rimosso dall'esecutore in BRANCH-DELETE-002 (PR #109), verificato da B
tramite inventari completi heads/branches main-only. Conservazione: storia contenuta in main alla revisione
`e7efb79a86e28209528ebcf4d85ca41f526b9b39`.
NODE aggiornato ad archived, flag archived_branch e viste rigenerate.
Nomi e padri Git sotto restano provenienza e contratti logici, non ref presenti.
Codice, API e collaudi applicativi conservano il loro ambito precedente.
Per la coda documentale e lo stato finale leggere CHECKPOINT.md radice e
il checkpoint BRANCH-CLEANUP-001-FINAL sulla PR della registrazione.


CPU-001 **implementata, verificata e integrata** in main.
Provider read-only stdlib: architettura piattaforma e numero logico
tramite `platform.machine` e `os.cpu_count` (possono essere unknown).
PR #25 foglia in hardware/cpu/main, #26 in hardware/main e #27 in main.
CI 38913317440, 38913438829, 38913533057 success, incluse 8 prove
CPU sintetiche e 12 hardware/RAM.
Collaudo HW-MANUAL-001 sulla macchina dell'utente ancora in attesa.
Non include core fisici certi, GPU, prestazioni o driver.
DATA-001: nessun inventario reale versionato.

## Collaudo HW-MANUAL-001 confermato (16 ottobre 2026)

L'utente ha eseguito Ruth sul commit `59b26fd6b427e6756ce0285fee60590cb077ebf1` verificato dopo
fast-forward della copia pulita. La fase CPU/RAM della sequenza
PY -> DEP -> DIAG -> HW ha avuto esito positivo con stato available.
CPU-001 e' quindi anche accettata personalmente per il perimetro
approvato; GPU e compatibilita' runtime non verificate.
Nessun valore hardware reale pubblicato (DATA-001).
