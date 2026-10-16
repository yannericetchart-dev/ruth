# Hardware: checkpoint HW-001

- HW-SCOPE-001 approvato il 16 ottobre 2026: CPU (architettura,
  processori logici) e RAM fisica totale, con valori `unknown`
  quando non disponibili; GPU fuori ambito.
- Contratti DeviceInfo, ResourceReport, HardwareReport in sola lettura;
  `hardware.detect` e `hardware.resources`.
- Ingresso ufficiale: `main.py` default esegue PY -> DEP -> DIAG -> HW,
  anche con CPU/RAM parzialmente sconosciute. Non confondere l'inventario
  con idoneita' a inferenza.
- Stato candidatura: implementation complete, verification not_run,
  integration not_merged, manual acceptance pending.
- Prossima azione: CI e mappe, promozioni foglia/padre/root,
  poi collaudo HW-MANUAL-001 sul checkout pulito aggiornato.
- REMOTE-001 e DATA-001: nessun checkout originale o dato reale in Git.
