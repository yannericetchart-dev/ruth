# Dipendenze: checkpoint DEP-001

**DEP-001 implementata, verificata e integrata tecnicamente in main.**
Perimetro DEP-SCOPE-001 approvato il 16 ottobre 2026.

- Ingressi: `environment.dependencies.plan(requirements, python_report)`,
  `environment.plan(requirements, project_root=...)` e CLI
  `main.py dependencies pip` dal virtualenv del progetto.
- Risultati: `DependencyPlan` e `PackageState`, `present_unverified`,
  `missing`, `incompatible`, `blocked`, `inspection_failed`;
  nessun controllo nativo o installazione.
- Provenienza storica metadata-only:
  `ruth_runtime/venv_packages.py`, commit `4ceea508025650ed64b33190834d0b5c85f69db2`.
- PR #12 in environment/dependencies/main:
  merge `4af2050e5fc59477f9ab854173bf915b7933d186`.
- PR #13 in environment/main:
  merge `c89c02ac0afa828b2c53ebd922a04331069ba527`.
- PR #15 in main:
  merge `a542f9ee9f6d09a0496bf756615d2295a7f59ce0`.
- Test: GitHub Actions 38895348468, 38895479549, 38895836703,
  38895924627 success; 14 Python, 12 dipendenze, 13 root/CLI, 237 mappe.
  Solo fixture sintetiche; presenza non equivale a compatibilita'.
- DEP-MANUAL-001: **active, collaudo utente non eseguito**. Prima sincronizzare
  in modo non distruttivo il clone pulito e confermare SHA; poi avviare
  `.venv/bin/python main.py dependencies pip` e confermare esito positivo
  del nuovo ingresso. Non avviare la migrazione successiva.
- DIAG-001, CUDA e installazioni fuori ambito (DATA-001);
  sviluppo remoto e sync post-merge secondo REMOTE-001.

## Chiusura DEP-MANUAL-001 (16 ottobre 2026)

Collaudo personale confermato dopo sincronizzazione fast-forward:
commit `e197cc86829acf52e326a712331cc7ff6cfef6b5`, comando `.venv/bin/python main.py dependencies pip`,
report `present_unverified` e riepilogo di compatibilita' non verificata.
Stato **complete**. Non sono pubblicati dettagli o log di host (DATA-001).
MAIN-FLOW-001 integra in seguito il controllo nel normale avvio sequenziale
senza argomenti, con una verifica minima di pip e test dedicati.

## Collegamento al flusso principale (MAIN-FLOW-001)

La verifica read-only dei metadati DEP-001 e' disponibile ora anche nel
flusso default, dopo Python, inizialmente sul solo `pip` (smoke check,
non lista requisiti complessivi). PR #17 integrata, CI `38900998138`
success. Il collaudo personale del nuovo flusso default e' pendente.

## MAIN-MANUAL-001 completata (16 ottobre 2026)

Avvio normale di Ruth (`main.py`, senza argomenti) confermato
personalmente sul commit `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`:
isolamento Python prima del controllo dei metadati del solo `pip`,
stato `present_unverified`, riepilogo finale positivo. Non implica
elenco completo dei requisiti o compatibilita' operativa.
