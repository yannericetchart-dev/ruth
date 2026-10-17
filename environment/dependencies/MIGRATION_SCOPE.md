# DEP-SCOPE-001: verifica delle dipendenze Python generiche

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


Perimetro funzionale **approvato esplicitamente dall'utente il 16 ottobre 2026** (MIG-FLOW-001).
Porzione DEP-001; fase 3. La revisione di origine e' main `6dfc34252501f9e708855588c69fbad7d39af73b`.
Sorgente storica: `ruth_runtime/venv_packages.py` al commit `4ceea508025650ed64b33190834d0b5c85f69db2`; solo ispezione dei metadati, senza lock CUDA, requisiti compilazione e backend nativi.

## Funzioni, ingressi e risultati

| Funzione | Ingresso | Risultato |
| --- | --- | --- |
| `environment.dependencies.plan` | Elenco esplicito di nomi, eventualmente `nome==versione`, e `PythonEnvironment` | `DependencyPlan` immutabile con `PackageState` per ogni pacchetto, stato ed eventuale versione rilevata. |
| `environment.plan` | Requisiti e radice del progetto | Controlla prima `environment.inspect`, poi pianifica tramite il componente; nessuna query se il virtualenv non e' conforme. |
| `DependencyPlan.to_dict` | Report in memoria | Dizionario transitorio, non scritto in Git. |
| `main.py dependencies pip [nome==versione ...]` | Argomenti CLI | Stampa avvio, ogni stato, motivi e risultato; codice 0 se tutti `present_unverified`, 1 per controlli non confermati, 2 per uso errato. |

Gli stati sono `present_unverified`, `missing`, `incompatible` (soltanto confronto di versione esatta), `blocked` e `inspection_failed`.
Il nome e' validato e i duplicati normalizzati sono rifiutati; intervalli e specifiche complesse non sono supportati in questa porzione. Il controllo Python gia' implementato resta il gate necessario: se non isolato, non viene interrogato alcun metadato di pacchetto.

## Effetti e limiti

Read-only con `importlib.metadata.version` del processo attuale: nessuna importazione del pacchetto, installazione, download, modifica della `.venv`, analisi GPU, build o lock. La presenza di un pacchetto *non* ne certifica compatibilita', integrita', sicurezza o operativita' runtime. Non salvare versioni rilevate, percorsi o dati host in Git (DATA-001).

La diagnostica avanzata DIAG-001, le versioni con intervalli, le dipendenze native e la configurazione backend restano in porzioni successive.
Il comando `python3 main.py` senza argomenti conserva la funzione e i messaggi di PY-001.

## Verifiche e sequenza

Test unitari con lookup sintetico per presente, assente, esatto compatibile/incompatibile, errori di lettura, duplicati, input malformati e gate Python; test CLI per esiti e codici; verifica delle mappe e CI remoto. Revisioni e promozioni `environment/dependencies/work/DEP-001` -> `environment/dependencies/main` -> `environment/main` -> `main` senza nuovo via libera (MIG-FLOW-001).
Il collaudo personale dopo merge e sincronizzazione locale (DEP-MANUAL-001) segue MIG-FLOW-002 e impedisce di iniziare il componente successivo prima della sua conferma.

**Stato:** codice candidato nel branch di lavoro, test della revisione finale non ancora eseguiti, non integrato in main.

## Stato di promozione (16 ottobre 2026)

- PR #12 verso environment/dependencies/main: merge `4af2050e5fc59477f9ab854173bf915b7933d186`.
- PR #13 verso environment/main: merge `c89c02ac0afa828b2c53ebd922a04331069ba527`.
- Verifica remota su fixture sintetiche e API/CLI: run 38895348468 e
  38895479549, success. Main e collaudo personale ancora pendenti.

## Integrazione finale DEP-001 (16 ottobre 2026)

- Il componente e' integrato in main tramite PR #15,
  merge `a542f9ee9f6d09a0496bf756615d2295a7f59ce0`.
- Sono integrati e revisionati `environment.dependencies.plan`,
  `environment.plan` e l'ingresso `main.py dependencies ...`;
  `main.py` senza argomenti conserva PY-001.
- CI GitHub 38895348468, 38895479549, 38895836703 e
  38895924627: success sui test sintetici, CLI e mappe.
- **DEP-MANUAL-001 in attesa**: sincronizzazione locale e avvio personale
  da parte dell'utente sulla revisione finale, come MIG-FLOW-002 richiede.
  Non e' avviata alcuna migrazione successiva.

## DEP-MANUAL-001 concluso (16 ottobre 2026)

Utente: sincronizzazione locale pulita tramite fast-forward e controllo
personale `dependencies pip` positivo sul commit `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
Esito limitato alla presenza di metadati (non compatibilita' runtime).
La nuova richiesta MAIN-FLOW-001 prevede l'esecuzione automatica PY-001
poi DEP-001 nel comando default `main.py`; la relativa implementazione e
il nuovo collaudo sono separati dal test mirato gia' accettato.

## MAIN-FLOW-001 in main (16 ottobre 2026)

Integrata via PR #17 l'esecuzione automatica `main.py`: Python prima,
quindi controllo minimale `pip`. I comandi espliciti di DEP-001
restano disponibili, e la loro accettazione manuale e' gia' completa.
Per la nuova sequenza il collaudo manuale MAIN-MANUAL-001 e' ancora
pendente e distinto da DEP-MANUAL-001.
