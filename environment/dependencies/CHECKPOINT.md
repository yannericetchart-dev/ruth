# Dipendenze: checkpoint DEP-001

DEP-SCOPE-001 approvato il 16 ottobre 2026. Ispezione pacchetti Python
in sola lettura, metadati generici, solo versioni esatte e controllo
bloccante dell'ambiente Python isolato. DIAG-001 e hardware fuori ambito.

- Origine storica: `ruth_runtime/venv_packages.py` al commit
  `4ceea508025650ed64b33190834d0b5c85f69db2` (sottoinsieme metadata-only).
- Implementazione: completa entro il perimetro approvato.
- Test: GitHub Actions 38895348468 e 38895479549 success,
  14 test Python, 12 dipendenze, 13 root/CLI, 237 mappe.
- Integrazione: PR #12 in environment/dependencies/main,
  merge `4af2050e5fc59477f9ab854173bf915b7933d186`;
  PR #13 in environment/main,
  merge `c89c02ac0afa828b2c53ebd922a04331069ba527`.
- Main finale: **ancora pendente**. Verifica nativa e collaudo
  personale DEP-MANUAL-001 non eseguiti. Non avviare migrazione seguente.
- Prossimo passo: PR environment/main -> main, test automatici,
  mappe/checkpoint finali, sincronizzazione non distruttiva del clone pulito
  e collaudo personale `.venv/bin/python main.py dependencies pip`.
- REMOTE-001, MIG-FLOW-001/002, DATA-001: rispettati; nessun dato host in Git.
