# PY-SCOPE-001: ispezione strutturale dell'ambiente Python

## Fonte operativa archiviata (17 ottobre 2026)

Fonte PY-001/DEP-001: tag annotato remoto verificato
`archive/2026-10-17/feat/gemma3-phase28-context-dispatch`, commit
`4ceea508025650ed64b33190834d0b5c85f69db2`; branch originario
`feat/gemma3-phase28-context-dispatch`. Prova del ref e oggetto tag in
BRANCH_ARCHIVE.md; stessa provenienza, nessun codice nuovo importato.
Il ramo storico Python `environment/python/main` e' archived/absent; tip storico
`b80520ad903ff18f93398725ea95cd91f51da5d8`, ora preservato anche da
`archive/2026-10-17/environment/python/main`. Rilascio utente confermato e
rimozione verificata in BRANCH-DELETE-002. Le promozioni e prove sotto sono storiche.


Stato della revisione funzionale: approvata dall'utente il 16 ottobre 2026, incluso l'ingresso ufficiale.
Porzione: PY-001. Regola di esecuzione: MIG-FLOW-001.

Il codice e' gia implementato in environment/python/main alla revisione
`b80520ad903ff18f93398725ea95cd91f51da5d8`, con revisione tecnica completata.
Il codice e' stato promosso in environment/main tramite PR #3 e in main
tramite PR #8. Questo documento conserva il perimetro approvato.

## Funzioni

| Funzione | Ingresso | Risultato e comportamento |
| --- | --- | --- |
| environment.python.inspect | Radice del progetto; osservazioni opzionali per test deterministici | Legge prefissi, versione dell'interprete e configurazione .venv/pyvenv.cfg. Restituisce PythonEnvironment con esito e motivi. |
| environment.inspect | Radice del progetto | API del sottosistema environment; inoltra al componente Python e restituisce lo stesso tipo di report. |
| runtime.inspect_environment | Radice del progetto | Ingresso comune di Ruth; usa l'API environment conservandone il contratto. |
| PythonEnvironment.to_dict | Report in memoria | Converte il report in un dizionario, senza salvarlo su disco. |

## Controlli eseguiti

- Esistenza e leggibilita della configurazione .venv/pyvenv.cfg.
- La .venv non deve essere un collegamento simbolico.
- Il prefisso dell'interprete deve coincidere con la .venv del progetto e
  differire dal prefisso del Python di base.
- La configurazione deve dichiarare una sola volta
  include-system-site-packages = false. Valori mancanti, non validi,
  ripetuti o malformati fanno fallire il controllo; true indica pacchetti
  di sistema condivisi e non supera il controllo.
- Versione Python almeno 3.10 per questa verifica generica.

## Report comune

PythonEnvironment e' una dataclass immutabile con:

- project_root: radice del progetto ricevuta.
- interpreter: interprete osservato.
- python_version: versione osservata.
- interpreter_in_project_venv: esito complessivo dei controlli indicati.
- venv_status: ok oppure invalid_or_inactive.
- reasons: motivi del mancato superamento, ad esempio configurazione
  assente/non valida, interprete esterno, versione insufficiente o errore di lettura.

I percorsi e i valori reali vengono rilevati soltanto durante l'esecuzione.
Non sono incorporati nel codice; eventuali salvataggi del report devono
restare locali ed esclusi da Git secondo DATA-001.

## Limiti ed effetti

L'ispezione e' in sola lettura. Non crea o modifica virtualenv, non installa
Python o pacchetti, non scarica modelli e non esegue inferenza o benchmark.
Non controlla versioni dei pacchetti, sys.path/PYTHONPATH, disponibilita GPU
o compatibilita dei backend. L'esito riguarda prefissi e configurazione del
virtualenv; non certifica l'intero ambiente di esecuzione di Ruth.

## Verifica e completamento dopo il verde

La revisione tecnica ha superato 14 test sintetici Python e 6 test delle
mappe, inclusa la catena reale delle tre API pubbliche. Le evidenze restano
locali. Durante le promozioni verificare compatibilita, test pertinenti,
portabilita dei file e coerenza delle mappe.

Il verde su questo perimetro autorizza il completamento da
environment/python/main a environment/main e quindi main, comprese le
correzioni necessarie entro il perimetro e gli aggiornamenti dei checkpoint,
senza ulteriori conferme intermedie. Nuovi dubbi su decisioni o un ampliamento
sostanziale del perimetro richiedono un chiarimento.

## Ingresso Ruth e accettazione personale (MIG-FLOW-002)

L'ingresso ufficiale `python3 main.py` esegue il controllo in sola lettura
e mostra avvio, stato e motivi. Codice integrato in main tramite PR #8.

GitHub Actions run 38885972247 e 38886317198: success per test sintetici,
catena API e ingresso Ruth e mappe; non equivalgono a verifica nativa.

**PY-MANUAL-001 completata il 16 ottobre 2026**. L'utente ha eseguito
personalmente Ruth sul checkout alla revisione
`ef4654af8ab106ef194af254d9f3884e5cabb6b0` e ha confermato il messaggio di ambiente conforme.
Non sono pubblicati dati host, log o credenziali secondo DATA-001.

Il prossimo componente richiede nuova approvazione funzionale MIG-FLOW-001.
