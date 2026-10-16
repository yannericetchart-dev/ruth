# Ruth: registro della migrazione

Lo scaffold non incorpora il codice applicativo storico. Ogni porzione deve
avere una fonte immutabile, una destinazione, test e una revisione separata.
I vecchi branch vengono conservati. Le misure e gli artefatti specifici di
macchina non sono candidati alla migrazione secondo DATA-001.

## Prima porzione: MIG-001 / PY-001

- Fonte: `feat/gemma3-phase28-context-dispatch`, commit
  `4ceea508025650ed64b33190834d0b5c85f69db2`,
  `ruth_runtime/venv_packages.py`, logica di verifica del virtualenv.
- Destinazione: `environment/python/runtime`; contratto pubblico del padre
  in `environment/runtime`; accesso comune dalla radice `runtime`.
- Ambito: ispezione in sola lettura dell'interprete e dell'isolamento della
  `.venv` del progetto. Nessuna installazione, download o importazione del
  lock CUDA specifico. Le versioni dei pacchetti sono una porzione successiva.
- Stato: integrata tramite PR #2 in `environment/python/main`, merge
  `8c76080df5ba34e354f484c92adf7e132b594c2d`, dopo verifica e revisione
  su fixture sintetiche. Ancora da promuovere a environment/main e main.
  Il codice e' estratto e adattato alla nuova API, non copiato integralmente.
- Differenze intenzionali: configurazioni vuote, duplicate o illeggibili
  falliscono il controllo, anziche essere considerate implicitamente isolate.
  Il contratto vive nel padre; la configurazione e' esplicita, senza root
  derivata dal vecchio launcher o dipendenze da lock e installer.
- Verifica prevista: fixture sintetiche per virtualenv attivo, errato,
  globale, con site-packages condivisi e configurazione illeggibile.
- Revisione PY-REV-001 completata: dichiarazioni malformate ripetute
  correttamente rifiutate e API pubbliche eseguite lungo la catena reale.
  Prima promozione PY-PROMOTE-001 completata nel branch del componente.
  L'albero del merge coincide con la revisione verificata; nessuna promozione
  nei branch superiori ancora eseguita.

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
