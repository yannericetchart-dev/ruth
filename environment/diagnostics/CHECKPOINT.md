# Diagnostica: checkpoint DIAG-001

DIAG-SCOPE-001 approvato dall'utente il 16 ottobre 2026.
Stato: **implementazione completa per questo perimetro; CI passed;
integrazione merged; collaudo personale ancora in attesa**.

- Contratto puro `DiagnosticIssue` / `DiagnosticReport` frozen;
  traduzione di report PY e DEP senza host probing.
- Ingresso standard `main.py`: PY -> DEP -> DIAG, con diagnostica
  disponibile anche se PY fallisce e DEP viene saltato.
- Promozioni: PR #20 -> environment/diagnostics/main
  (`ec7eba74bc71e62ea8562fa51d62295f5f3c01b8`),
  PR #21 -> environment/main
  (`69ae0821cc0043bd76254bc46dff12b40d0d7b30`),
  PR #22 -> main (`dfdb41e981f32ee233320f77998aadab55ce4d7b`).
- Test: CI GitHub 38908192677, 38908269925, 38908370200 success,
  13 test sintetici di DIAG, 14 PY, 12 DEP, 15 root/CLI e mappe.
  Nessuna verifica GPU/CUDA/log nativi.
- **DIAG-MANUAL-001 active**: avvio personale Ruth dalla copia
  aggiornata main, conferma SHA, stampa delle fasi e riepilogo.
- Dopo collaudo positivo: aggiornare stato in Git; poi il prossimo
  componente richiede un nuovo perimetro funzionale e approvazione.
- DATA-001, REMOTE-001 e MIG-FLOW-001/002 continuano a valere.
