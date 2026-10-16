# Python: checkpoint PY-001

Stato: prima porzione implementata, verificata e revisionata, integrata in
`environment/python/main` tramite PR #2. Non promossa a environment/main o main.
Contratto sperimentale. Fonte e perimetro: MIGRATION.md alla radice.

## Implementato

- `environment.python.inspect(Path)` ispeziona interprete, prefissi e
  configurazione `.venv/pyvenv.cfg`, senza scritture o installazioni.
- Il contratto PythonEnvironment appartiene al padre environment. La stessa
  risposta attraversa environment.inspect e runtime.inspect_environment.
- Configurazioni mancanti, illeggibili, ambigue, condivise o con virtualenv
  esterno/symlink non superano il controllo. Python minimo: 3.10, come nella
  logica generica di origine; non e' un requisito di compilazione CUDA.

## Verificato

Suite PY-001 su fixture sintetiche: virtualenv valido, interprete globale,
virtualenv esterno, site-packages condivisi, configurazioni mancanti/errate,
errori di lettura e codifica, versione minima, symlink, contratto dei padri
e assenza di modifiche alla configurazione.

## Revisione PY-REV-001

La revisione ha riprodotto e corretto un errore nella lettura delle
dichiarazioni ripetute: una riga malformata dopo una impostazione valida
veniva ignorata. Tutte le dichiarazioni della chiave di isolamento vengono
ora considerate; una configurazione ambigua fallisce il controllo.

Il contratto e' verificato anche eseguendo i tre ingressi pubblici reali
su osservazioni sintetiche, senza sostituire il componente figlio con un mock.
Verificato inoltre che una dichiarazione `passed` senza ambito esplicito
venga rifiutata dal validatore delle mappe.

Esito: nessun problema bloccante residuo nell'ambito di questa porzione.
Riferimento delle evidenze locali: PY-REV-001. La revisione non certifica
pacchetti, GPU o inferenza. Non e' una revisione indipendente di un secondo
manutentore.

Comandi dalla radice:

```sh
python3 -m unittest discover -s environment/python/tests -v
python3 -m unittest discover -s tests -v
python3 tools/maps.py --check
```

I risultati restano in `.local/verification/`. La presenza di Python non
dimostra dipendenze installate, backend disponibile o inferenza funzionante.
Non sono stati eseguiti benchmark o collaudi hardware.

## Prima promozione PY-PROMOTE-001

PR #2 integrata con merge commit `8c76080df5ba34e354f484c92adf7e132b594c2d`,
conservando la revisione esaminata `80218739327666b0250c3715ba9cfb6fe1f114d5`.
L'albero risultante dal merge e' identico alla revisione che ha superato i
test sintetici e il controllo delle mappe. Evidenze locali: PY-PROMOTE-001.

Il campo integration del nodo resta not_merged: il suo padre dichiarato e'
environment/main, dove questa porzione non e' ancora presente. La promozione
dal branch di lavoro al branch canonico del componente e' invece completata.

## Prossima azione

Preparare e verificare la promozione da environment/python/main a
environment/main. La promozione finale a main e' un passaggio successivo,
con checkpoint aggiornati ad ogni passaggio. In seguito pianificare le dipendenze
generiche: non copiare i lock nativi specifici dalla fonte storica.
