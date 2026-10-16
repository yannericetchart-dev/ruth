# Ruth: registro della migrazione

Lo scaffold non incorpora il codice applicativo storico. Ogni porzione deve
avere una fonte immutabile, una destinazione, test e una revisione separata.
I vecchi branch vengono conservati. Le misure e gli artefatti specifici di
macchina non sono candidati alla migrazione secondo DATA-001.

## Prima porzione prevista: MIG-001 / PY-001

Revisione funzionale: [PY-SCOPE-001](environment/python/MIGRATION_SCOPE.md),
approvata dall'utente il 16 ottobre 2026. Completare la
porzione fino a main senza ulteriori conferme intermedie (MIG-FLOW-001).

- Fonte: `feat/gemma3-phase28-context-dispatch`, commit
  `4ceea508025650ed64b33190834d0b5c85f69db2`,
  `ruth_runtime/venv_packages.py`, logica di verifica del virtualenv.
- Destinazione: `environment/python/runtime`; contratto pubblico del padre
  in `environment/runtime`; accesso comune dalla radice `runtime`.
- Ambito: ispezione in sola lettura dell'interprete e dell'isolamento della
  `.venv` del progetto. Nessuna installazione, download o importazione del
  lock CUDA specifico. Le versioni dei pacchetti sono una porzione successiva.
- Stato: prima implementazione revisionata (PY-REV-001) e integrata tramite
  [PR #2](https://github.com/yannericetchart-dev/ruth/pull/2), branch
  `environment/python/main`, con stato aggiornato al commit
  `b80520ad903ff18f93398725ea95cd91f51da5d8`. Non promossa ai padri superiori;
  il codice di questo branch main resta lo scaffold.
- Esito della revisione: corretto il caso delle dichiarazioni malformate
  ripetute e verificata la catena reale delle API. Prima promozione
  PY-PROMOTE-001 completata, merge `8c76080df5ba34e354f484c92adf7e132b594c2d`.
  L'albero del merge coincide con la revisione verificata. Da promuovere a
  environment/main e in seguito main.
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

## Stato PY-001 (16 ottobre 2026)

- Perimetro PY-SCOPE-001 approvato; ispezione Python in sola lettura e ingresso `python3 main.py`.
- Promozione foglia: PR #2 in environment/python/main, commit b80520ad903ff18f93398725ea95cd91f51da5d8.
- Promozione intermedia: PR #3, merge environment/main fd61723874052053f38446ee82eca954870804e1.
- Integrazione principale: PR #8, merge main ce1b0bb47b4ae4ad0da40f3f83c964b66015680f.
- Test su GitHub superati: run 38885972247 e 38886317198 (Python sintetico, API, ingresso ufficiale, mappe). Nessun collaudo macchina nativa.
- `local_sync: blocked_divergence` per il checkout principale osservato su branch di lavoro con modifiche; non sovrascrivere nulla. Il collaudo personale PY-MANUAL-001 e' in attesa della sincronizzazione sicura e della conferma positiva.
