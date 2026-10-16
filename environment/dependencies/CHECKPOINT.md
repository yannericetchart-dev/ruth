# Dipendenze: checkpoint DEP-001

- DEP-SCOPE-001 approvato il 16 ottobre 2026: metadati Python generici,
  versioni esatte, stati e output CLI in sola lettura.
- Branch canonico foglia: `environment/dependencies/main` (gia' presente).
  Lavoro: `environment/dependencies/work/DEP-001`.
- Sorgente storica: `ruth_runtime/venv_packages.py` al commit
  `4ceea508025650ed64b33190834d0b5c85f69db2`.
- Codice implementato sul ramo di lavoro, **non ancora verificato o integrato**
  in foglia, padre o main. Stato not_run.
- Controlli: `environment.dependencies.plan`, `environment.plan`,
  `main.py dependencies ...`; CLI Python senza argomenti preservata.
- Limiti: nessuna installazione, lock CUDA, validazione backend, diagnostica
  avanzata, intervalli di versione o dati host nel repository.
- Prossima azione: test CI sintetici, rigenerazione mappe, review e promozioni
  fino a main; poi sincronizzazione remota/locale sicura e collaudo utente
  DEP-MANUAL-001. REMOTE-001, MIG-FLOW-001/002 e DATA-001 vigenti.
