# Environment: checkpoint ENV-001

Lavoro attivo: PY-001, branch `environment/python/work/inspect-venv`.
La sola ispezione Python e' implementata, verificata e revisionata su fixture sintetiche,
ma non integrata nel branch environment/main. Vedere
[checkpoint del componente](python/CHECKPOINT.md).

API comune disponibile nel branch di lavoro: `environment.inspect(Path)`
restituisce PythonEnvironment, identico al risultato del figlio.
Pianificazione dipendenze, installazioni e diagnostica estesa restano proposte.
Le mappe della radice riflettono questo avanzamento limitato.

PY-REV-001 completata: corretta la gestione delle dichiarazioni malformate
ripetute e verificata la catena reale delle API pubbliche. Prossima azione:
prima promozione della PR #2 a environment/python/main; nessun merge eseguito.
