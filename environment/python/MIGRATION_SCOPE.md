# PY-SCOPE-001: ispezione strutturale dell'ambiente Python

Stato della revisione funzionale: approvata dall'utente il 16 ottobre 2026, incluso l'ingresso ufficiale.
Porzione: PY-001. Regola di esecuzione: MIG-FLOW-001.

Il codice e' gia implementato in environment/python/main alla revisione
`b80520ad903ff18f93398725ea95cd91f51da5d8`, con revisione tecnica completata.
Non e' ancora presente in environment/main o main. Questo documento descrive
il perimetro da revisionare prima di completarne le promozioni.

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

## Ingresso Ruth e collaudo personale richiesto (MIG-FLOW-002)

Nella versione integrata in main il punto d'ingresso ufficiale di Ruth deve
eseguire questa ispezione e mostrare messaggi a terminale relativi all'avvio
del controllo, al suo esito e ai motivi rilevanti. Il collegamento al punto
d'ingresso e i messaggi non sono ancora implementati: sono un requisito
del completamento di questa porzione.

Dopo l'integrazione fornire all'utente il comando concreto per avviare Ruth
e gli esiti attesi. L'utente lo esegue localmente di persona e conferma
esplicitamente l'esito positivo prima di iniziare la migrazione successiva.
Registrare revisione provata e stato del collaudo nel checkpoint; mantenere
log e risultati reali esclusivamente locali. PY-MANUAL-001 e' pianificata;
non e' stato eseguito o confermato un collaudo personale.

Stato implementazione dell'ingresso: candidato in PR #3, non ancora integrato in main. Verifica sulla revisione remota e collaudo utente ancora necessari.

Verifica: GitHub Actions run 38885972247 superato; primo merge in environment/main `fd61723874052053f38446ee82eca954870804e1`. Promozione in main e collaudo personale ancora pendenti.
