# Hardware: checkpoint HW-001

## Stato del ref remoto (17 ottobre 2026)

Branch logico `hardware/main`: **archived / ref remoto assente**.
Tip prima della rimozione: `60e137a422403efa7d8d4005240abf317ce87e24`.
Rimosso dall'esecutore in BRANCH-DELETE-002 (PR #109), verificato da B
tramite inventari completi heads/branches main-only. Conservazione: storia contenuta in main alla revisione
`e7efb79a86e28209528ebcf4d85ca41f526b9b39`.
NODE aggiornato ad archived, flag archived_branch e viste rigenerate.
Nomi e padri Git sotto restano provenienza e contratti logici, non ref presenti.
Codice, API e collaudi applicativi conservano il loro ambito precedente.
Per la coda documentale e lo stato finale leggere CHECKPOINT.md radice e
il checkpoint BRANCH-CLEANUP-001-FINAL sulla PR della registrazione.


**HW-001 implementata, verificata, integrata in main e accettata
personalmente dall'utente** (16 ottobre 2026).

- HW-SCOPE-001: CPU architettura/processori logici e RAM fisica
  totale tramite API stdlib, stato available/partial/unknown; nessuna GPU.
- Contratti frozen: `DeviceInfo`, `HardwareReport`,
  `ResourceReport`, accessibili via `hardware.detect()` e
  `hardware.resources()`.
- Ingresso ufficiale `main.py`: PY -> DEP -> DIAG -> HW;
  inventario hardware indipendente dal risultato Python.
- PR #25 -> hardware/cpu/main, PR #26 -> hardware/main,
  PR #27 -> main, merge `35a6ac79a1b1287090c1a3d1dc9208d4129a15ec`.
- CI GitHub runs 38913317440, 38913438829 e 38913533057:
  success, 74 test sintetici e 249 mappe coerenti.
- **HW-MANUAL-001: complete**. L'utente ha verificato il commit
  `59b26fd6b427e6756ce0285fee60590cb077ebf1` nella copia pulita aggiornata in fast-forward,
  avviato personalmente `.venv/bin/python main.py` e confermato
  i quattro passaggi e CPU/RAM rilevate con stato available.
- Non certifica GPU, backend, RAM libera o inferenza.
  Nessun numero o inventario reale della macchina salvato in Git (DATA-001).
- Componenti GPU o successivi soltanto dopo un nuovo perimetro approvato
  (MIG-FLOW-001); il vecchio checkout locale resta preservato (REMOTE-001).

## GPU-SCOPE-001: candidatura NVIDIA (16 ottobre 2026)

Perimetro NVIDIA approvato, runtime GPU vendor-neutral candidato
senza modificare il rilevamento CPU/RAM integrato; CI e promozioni
ancora pendenti, AMD/Intel inattive. GPU-MANUAL-001 prevista dopo
main, senza dati host su Git.

## GPU-001 NVIDIA: integrazione verificata (16 ottobre 2026)

- PR #30 -> hardware/gpu/nvidia/main, #31 -> hardware/gpu/main,
  #32 -> hardware/main, #33 -> main (merge
  `54a02cf37262a78f5a1ed0aa84cd6d210060a14c`).
- CI 38918374871, 38918650589, 38918759116 e 38918867451:
  success, 92 test sintetici e 258 mappe coerenti. Nessun test GPU
  fisico o CUDA e' certificato; AMD/Intel non implementate.
- `main.py`: PY->DEP->DIAG->HW CPU/RAM->NVIDIA.
- Stato tecnico: verificato e integrato; GPU-MANUAL-001 ancora
  in attesa di conferma personale post-sincronizzazione.

## GPU-MANUAL-001 completata (16 ottobre 2026)

L'utente ha provato personalmente sul commit `c344c71013cfa05da4f49f3204900f4bdc89b8ea`
l'avvio `main.py` completo e confermato la quinta fase NVIDIA
con stato `available` e riepilogo positivo. Il blocco di collaudo
GPU-001 e' chiuso; le note precedenti di stato pending sono storiche.
Nessun modello, misura di VRAM o inventario reale viene pubblicato
su Git (DATA-001); driver, CUDA e inferenza non sono certificati.
