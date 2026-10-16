# DIAG-SCOPE-001: diagnostica read-only dell'ambiente

- Stato del perimetro: **approvato** dall'utente il 16 ottobre 2026.
- Fonte canonica della revisione iniziale: `main`
  `b2bc0095407edf766c2563bbedc2e69889b074ce`.
- Scopo: interpretare e spiegare i report gia' prodotti da PY-001 e
  DEP-001. Implementazione nuova e portabile, non migrazione di log CUDA.
- API: `environment.diagnostics.inspect(python_report, dependencies=None)`
  restituisce `DiagnosticReport(status, issues)` e
  `DiagnosticIssue(code, severity, message, suggested_action, package)`,
  frozen dataclasses e `to_dict` transitorio.
- Stati: `ok_unverified`, `attention_required`,
  `inspection_incomplete`; il successo non certifica runtime o sicurezza.
- Diagnosi: .venv non conforme (configurazione, versione Python,
  interprete/isolamento), distribuzioni `missing`, `incompatible`,
  `blocked`, `inspection_failed`; categorie sconosciute non dedotte.
- Effetti: nessuna installazione, scansione della macchina, log, GPU,
  import di pacchetti o raccolta telemetria. Il report usa solo gli oggetti
  PY-001 e DEP-001 gia' disponibili, senza salvare dati (DATA-001).
- Default `main.py`: PY-001 poi DEP-001 poi DIAG-001. Se la fase PY
  fallisce, DEP non parte ma DIAG spiega il report Python; i sottocomandi
  mirati rimangono disponibili e non richiedono una nuova approvazione.
- Test: fixture sintetiche per ragioni note e non note, package states,
  report incoerenti, esito e ordine dei passaggi, CLI e mappe. CI remota
  e poi integrazione
  `environment/diagnostics/main` -> `environment/main` -> `main`.
- Collaudo personale DIAG-MANUAL-001 dopo merge e sincronizzazione
  non distruttiva, con commit della revisione provata e messaggi di Ruth.
- Non inclusi: CUDA, GPU, driver, log nativi, installazioni, riparazioni,
  inferenza e inventari specifici della macchina. Kaggle inattivo.

**Stato implementazione (16 ottobre 2026)**: DIAG-001 implementata,
verificata da CI e integrata in main tramite PR #20, #21 e #22,
merge finale `dfdb41e981f32ee233320f77998aadab55ce4d7b`.
Il collaudo personale DIAG-MANUAL-001 e' stato completato con esito
positivo sul commit `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb` (16 ottobre 2026), confermando
la sequenza PY -> DEP -> DIAG e lo stato `ok_unverified`.
Le evidenze di verifica riguardano fixture sintetiche, non la macchina
dell'utente ne' runtime nativi.
