# Python: checkpoint PY-001

Stato: prima porzione implementata e verificata nel branch di lavoro
`environment/python/work/inspect-venv`; non integrata nei branch padri.
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

Comandi dalla radice:

```sh
python3 -m unittest discover -s environment/python/tests -v
python3 -m unittest discover -s tests -v
python3 tools/maps.py --check
```

I risultati restano in `.local/verification/`. La presenza di Python non
dimostra dipendenze installate, backend disponibile o inferenza funzionante.
Non sono stati eseguiti benchmark o collaudi hardware.

## Prossima azione

Revisionare questa porzione prima della promozione a environment/python/main,
poi environment/main e infine main. In seguito pianificare le dipendenze
generiche: non copiare i lock nativi specifici dalla fonte storica.
