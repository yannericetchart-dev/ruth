# Ruth: registro della migrazione

Lo scaffold non incorpora il codice applicativo storico. Ogni porzione deve
avere una fonte immutabile, una destinazione, test e una revisione separata.
I vecchi branch vengono conservati. Le misure e gli artefatti specifici di
macchina non sono candidati alla migrazione secondo DATA-001.

## Prima porzione prevista: MIG-001 / PY-001

- Fonte: `feat/gemma3-phase28-context-dispatch`, commit
  `4ceea508025650ed64b33190834d0b5c85f69db2`,
  `ruth_runtime/venv_packages.py`, logica di verifica del virtualenv.
- Destinazione: `environment/python/runtime`; contratto pubblico del padre
  in `environment/runtime`; accesso comune dalla radice `runtime`.
- Ambito: ispezione in sola lettura dell'interprete e dell'isolamento della
  `.venv` del progetto. Nessuna installazione, download o importazione del
  lock CUDA specifico. Le versioni dei pacchetti sono una porzione successiva.
- Stato: pianificata; nessun codice applicativo ancora importato.
- Verifica prevista: fixture sintetiche per virtualenv attivo, errato,
  globale, con site-packages condivisi e configurazione illeggibile.

## Porzioni successive

1. Dipendenze generiche e diagnostica dell'ambiente, con requisiti verificati.
2. Inventario hardware, separato dall'abilitazione del backend.
3. Catalogo e lettura GGUF, poi acquisizione Hugging Face.
4. Backend CUDA e llama.cpp/llama-cpp-python dietro la sessione comune.
5. Profilazione con schemi portabili e risultati esclusivamente locali.
6. Conversazione e terminale dal branch storico minimal-chat.

La provenienza puntuale e i conflitti fra implementazioni si risolvono prima
di ogni porzione; non dedurre il comportamento dai nomi dei branch. Le API
proposte possono cambiare con decisione registrata e aggiornamento dei padri.
