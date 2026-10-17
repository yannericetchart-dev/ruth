# Ruth: registro delle decisioni

Le decisioni sono riferite al progetto, senza informazioni personali o
inventari di macchina. Stati: proposed, accepted, superseded, rejected.
Ogni sostituzione deve collegare entrambe le decisioni e aggiornare le mappe.

## CONT-001: continuita' fra chat

- Stato: accepted.
- Decisione: GitHub e' il punto di ingresso comune per ChatGPT e Codex.
  Istruzioni persistenti impongono lettura di indice, decisioni, checkpoint
  e mappe, seguita da confronto con il codice realmente disponibile.
- Motivazione: evitare perdita di strategia e confusione fra proposto,
  implementato, verificato e integrato quando cambia la conversazione.
- Verifica richiesta: ripartenza senza cronologia su entrambi i client.
- Implementazione del protocollo: configurata e collaudata con ripartenza
  indipendente su entrambi i client. I risultati restano locali. Il collaudo
  riguarda la revisione di ingresso del protocollo, non tutte le chat future.
- Sostituisce: nessuna decisione registrata.

## MIG-FLOW-001: approvazione delle funzioni, poi integrazione completa

- Stato: accepted.
- Decisione: prima di ogni porzione di migrazione descrivere le funzioni
  all'utente e attendere il semaforo verde sul perimetro. Dopo l'approvazione
  procedere autonomamente attraverso test, revisioni e branch padri fino a
  main. Interrompere per dubbi o domande che richiedono una decisione, senza
  chiedere nuovamente conferma a ogni promozione ordinaria.
- Motivazione: consentire la revisione funzionale prima del lavoro e
  completare poi l'intera porzione approvata senza fermate intermedie.
- Applicazione completata: PY-001 approvata, integrata fino a main e
  collaudata positivamente dall'utente sul commit ef4654af8ab106ef194af254d9f3884e5cabb6b0.
  Il perimetro del prossimo componente richiede una nuova approvazione.
- Precisa: PLAN-001; la sequenza per componenti resta valida.
- Sostituisce: l'interpretazione operativa che richiedeva un nuovo via libera
  per ogni singolo passaggio fra branch padri.

## MIG-FLOW-002: esecuzione dall'ingresso e conferma manuale

- Stato: accepted.
- Decisione (revisione 2, 17 ottobre 2026): ogni porzione integrata in main
  deve essere eseguita dall'ingresso ufficiale Ruth. Dopo l'integrazione,
  l'utente avvia personalmente il flusso: un output di collaudo positivo
  incollato dall'utente costituisce gia' la conferma personale sufficiente,
  **senza chiedere un ulteriore "si"**. Questa interpretazione sostituisce
  il precedente requisito di una seconda approvazione separata.
- Verifica: confronto commit della prova con il ref main risolto su GitHub;
  presenza dell'intera sequenza, riepilogo finale positivo e, quando inclusi
  nel collaudo definito per lo scope, test mirati con esito "OK".
  Nessun fallimento o contraddizione. Prova incompleta o non congrua:
  non accettare e richiedere solo la parte mancante.
  CI e azioni dell'agente non sostituiscono il collaudo personale.
  Dopo la prova positiva, completare nello stesso turno checkpoint,
  NODE, mappe, PR, CI, merge in main e proporre lo scope successivo
  **senza scrivere nuovo codice prima della nuova approvazione**.
- Dati: log e risultati reali restano locali secondo DATA-001; in Git si
  registra solo lo stato del processo senza dati specifici di macchina.
- Estende: MIG-FLOW-001. Il verde iniziale autorizza il completamento fino
  a main; il collaudo personale positivo abilita la migrazione successiva.
- Ambito: tutte le porzioni, inclusa PY-001. Questa modifica registra la
  regola; non approva il perimetro corrente e non avvia migrazioni.

## DATA-001: repository portabile, artefatti locali

- Stato: accepted.
- Decisione: solo codice e materiale di progetto generico possono entrare
  in Git. Dati reali dipendenti da host o utente, log, misure, cache, build
  e modelli scaricati restano esclusi. Directory convenzionale: `.local/`.
- Motivazione: mantenere il progetto riutilizzabile su macchine differenti.
- Ambito: nuovo scaffold e migrazione; gli artefatti storici non vengono
  copiati nel nuovo main. La storia esistente resta conservata.
- Verifica richiesta: controllo dei percorsi e del contenuto prima di
  pubblicare ciascuna modifica.
- Sostituisce: l'ipotesi di migrare in Git le misure native dei vecchi branch.

## SCOPE-001: Kaggle inattivo

- Stato: accepted.
- Decisione: Kaggle resta previsto, senza sviluppo o attivazione corrente.
  Il codice storico viene censito e conservato nella sua fonte originaria.
- Motivazione: concentrare l'acquisizione iniziale su Hugging Face.
- Sostituisce: l'inclusione di Kaggle nel perimetro attivo della prima bozza.

## PLAN-001: ordine di esecuzione

- Stato: accepted.
- Decisione: (1) configurazione e collaudo dei client; (2) scaffold senza
  migrazione applicativa; (3) migrazione verificata per componente.
- L'albero e le firme proposte restano una base evolvibile, non una
  dichiarazione di capacita' gia' disponibili.
- Sostituisce: nessuna decisione registrata.

## SYNC-001: sincronizzare prima di riprendere o chiudere il lavoro

- Stato: superseded.
- Sostituita da: REMOTE-001, approvata il 16 ottobre 2026.
- Nota storica: il blocco di ripresa fino alla sincronizzazione del clone
  e' stato superato; non applicarlo al lavoro remoto via GitHub.
- Decisione: GitHub e checkout locale sono stati distinti. Prima della
  ripresa di un'attivita' interrotta e prima di dichiarare completo il lavoro
  dipendente dal checkout, verificare l'allineamento del locale con main.
- In assenza di accesso locale, l'agente deve dare **subito**, prima della
  prosecuzione o della chiusura dell'attivita', la procedura sicura con lo
  SHA target e richiedere all'utente il riscontro di sincronizzazione.
  Non basta offrire istruzioni solo al termine. Fino al riscontro registrare
  `local_sync: pending_no_access`; in caso di divergenze non risolte,
  `local_sync: blocked_divergence`.
- Se il checkout e' disponibile, sincronizzare solo con operazioni sicure,
  non distruttive, verificando branch, modifiche locali, remote e SHA finale.
- Il blocco di sincronizzazione non costituisce un nuovo semaforo verde
  funzionale: dopo la verifica il lavoro gia' approvato prosegue fino al merge
  in main secondo MIG-FLOW-001. Il collaudo personale rimane distinto e
  obbligatorio secondo MIG-FLOW-002.
- Scelta operativa: in ChatGPT per Ruth usare Chat + GitHub; non usare Work
  come soluzione automatica per l'accesso locale, per preservare la quota
  condivisa con Codex.
- Motivazione: evitare di lasciare non sincronizzata la macchina alla
  ripresa, oltre a non dichiarare completato un processo senza verificarla.
- Ambito: ChatGPT e Codex, con procedura pratica in PROJECT_SETUP.md.
- Approvazione: regola approvata il 16 ottobre 2026 e precisata esplicitamente
  dall'utente nella stessa data.
- Verifica richiesta: riscontro di un checkout autorizzato sul commit main
  e collaudo indipendente della ripresa; evidenze locali secondo DATA-001.
- Precisa: la formulazione iniziale di SYNC-001, che richiedeva istruzioni
  solo dopo il merge e permetteva `pending_no_access` senza un blocco
  alla ripresa dell'attivita'.

## REMOTE-001: sviluppo GitHub-first e sincronizzazione locale post-merge

- Stato: accepted.
- Decisione: ChatGPT e Codex modificano file versionati, decisioni,
  checkpoint e mappe *direttamente su GitHub*, usando branch, commit, PR
  e promozioni fino a main; il checkout principale locale resta intatto
  durante tutto lo sviluppo. Una modifica non pubblicata al clone non e'
  una fonte canonica e non viene importata o cancellata implicitamente.
- Test ed esecuzioni locali: Codex puo' creare copie temporanee pulite della
  revisione GitHub candidata **fuori dalla directory versionata di lavoro**
  o in `.local/verification/`, ignorata da Git. Sono ammessi file e build
  generati solo in tali spazi isolati, senza alterare il checkout principale.
  Le modifiche da integrare vanno pubblicate su GitHub e nuovamente verificate.
- Sincronizzazione: solo dopo il merge definitivo in main, confrontare
  commit GitHub e clone locale, aggiornare in sicurezza o dare all'utente
  i comandi e richiedere conferma. Un checkout sporco, divergente o su un
  branch diverso va preservato senza scarti automatici.
- Se GitHub non consente letture/scritture richieste, fermarsi ed esplicitare
  il blocco, senza convertire il flusso in sviluppo locale del clone.
- L'assenza di sync preliminare non blocca lo sviluppo remoto approvato.
  Restano MIG-FLOW-001 (merge completo senza conferme intermedie) e
  MIG-FLOW-002 (esecuzione dal punto d'ingresso e collaudo personale positivo).
- Motivazione: evitare le modifiche concorrenti non committate che rendono
  difficile la ripresa tra ChatGPT e Codex e separare i test temporanei
  dai file che verranno sincronizzati.
- Sostituisce: SYNC-001, le cui istruzioni di sync obbligatoria prima della
  ripresa sono ormai storiche e non vigenti.
- Ambito: entrambi i client; non autorizza la cancellazione o il recupero
  automatico di modifiche locali preesistenti. DATA-001 resta valido.
- Approvazione: richiesta esplicita dell'utente il 16 ottobre 2026.
- Verifica richiesta: collaudo indipendente di ripresa su ChatGPT e Codex,
  conferma che l'area di test e' ignorata e il checkout principale rimane
  invariato, verifica della promozione su GitHub e sincronizzazione finale.

## DEP-ORDER-001: dipendenze generiche prima dell'hardware

- Stato: accepted.
- Approvazione: semaforo verde dell'utente su DEP-SCOPE-001 il 16 ottobre 2026.
- Decisione: dopo PY-001 migrare solo la verifica read-only delle dipendenze
  Python generiche, con output dall'ingresso Ruth; poi diagnostica e hardware
  in porzioni successive soggette a perimetro approvato.
- Motivo: seguire il checkpoint e MIGRATION.md correnti, riallineando
  ARCHITECTURE.md. La diagnostica DIAG-001 non e' approvata in DEP-001.
- Vincoli: MIG-FLOW-001/002, REMOTE-001, DATA-001 e Kaggle inattivo.

## MAIN-FLOW-001: eseguire tutti i passaggi integrati da main.py in ordine

- Stato: accepted.
- Approvazione: chiarimento esplicito dell'utente il 16 ottobre 2026,
  contestuale al collaudo DEP-MANUAL-001 completato.
- Decisione: `main.py` senza argomenti avvia e coordina in sequenza
  tutte le operazioni gia' migrate, approvate e integrate, rispettando i
  prerequisiti. Stampa stato e motivi, arresta le operazioni dipendenti se
  manca il prerequisito; i sottocomandi restano per diagnosi mirate.
- Primo ambito: controllo isolamento Python (PY-001), poi verifica minima
  dei metadati del pacchetto `pip` (DEP-001), senza presentare `pip`
  come lista esaustiva dei requisiti di Ruth. Ulteriori requisiti e le
  future funzioni si collegano solo nei perimetri autorizzati.
- Verifica richiesta: test sintetici dell'ordine e del mancato avvio
  delle dipendenze in caso di Python non conforme, CI GitHub, avvio reale
  da parte dell'utente dopo il merge in main.
- Ambito: root/orchestrazione, applicabile a ChatGPT e Codex;
  precisa MIG-FLOW-002; compatibile con MIG-FLOW-001, REMOTE-001 e DATA-001.

- Stato applicativo MAIN-FLOW-001 (16 ottobre 2026): codice e verifiche
  automatiche integrate in main tramite PR #17, collaudo personale
  MAIN-MANUAL-001 superato sul commit
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
  L'accettazione DEP-MANUAL-001 resta distinta e confermata.
  Nessun componente successivo avviato senza nuovo perimetro approvato.

## DIAG-SCOPE-001: diagnosi dai soli report ambiente esistenti

- Stato: accepted.
- Approvazione: esplicita dell'utente il 16 ottobre 2026.
- Scopo: convertire gli esiti di Python e delle dipendenze in codici,
  spiegazioni e suggerimenti portabili, senza modificare l'ambiente.
- Ingresso: report PY-001 e opzionale DEP-001; uscita DiagnosticReport
  immutabile. Non inferire problemi CUDA/GPU dai report attuali.
- Integrazione: il flusso ordinario `main.py` segue PY -> DEP -> DIAG;
  se PY fallisce DEP e' saltato ma DIAG spiega PY. Esporre sempre
  esiti ed eventuali blocchi all'utente (MAIN-FLOW-001).
- Sicurezza: suggerimenti e non riparazioni; dati host esclusi (DATA-001).
- Sequenza: chiudere diagnostica prima dell'hardware, correggendo l'ordine
  di ARCHITECTURE.md; nessun hardware migrato implicitamente.
- Test: fixture sintetiche e ordine/ragioni, mappe e CI GitHub;
  collaudo personale DIAG-MANUAL-001 post-merge (MIG-FLOW-002).

- Applicazione DIAG-SCOPE-001: implementata e verificata in GitHub,
  integrata in main mediante PR #22 il 16 ottobre 2026.
  Collaudo personale DIAG-MANUAL-001 **completato** sulla revisione
  `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb` il 16 ottobre 2026, confermati ordine e
  riepilogo. Ogni ulteriore migrazione richiede nuova approvazione.

## HW-SCOPE-001: primo inventario CPU e RAM, GPU successiva

- Stato: accepted; approvazione dell'utente il 16 ottobre 2026.
- Scopo: report hardware portabile e read-only per CPU logiche,
  architettura e RAM fisica totale osservabile, stati unknown/partial.
- Contratti: DeviceInfo, ResourceReport, HardwareReport; API
  `hardware.cpu.detect`, `hardware.detect`, `hardware.resources`.
- Ingresso: `main.py` automatico continua le fasi precedenti e aggiunge
  HW come quarto passaggio indipendente, anche se Python fallisce.
- Limiti: niente GPU vendor, core fisici certi, RAM libera,
  benchmark, driver, CUDA, inference, installazioni o risultati reali Git.
- Verifica: CI sintetica, mappe e promozioni fino a main; collaudo personale
  HW-MANUAL-001 necessario prima del successivo componente.
- Vincoli: MIG-FLOW-001/002, MAIN-FLOW-001, REMOTE-001, DATA-001.

- Applicazione HW-SCOPE-001 (16 ottobre 2026): integrata in main
  tramite PR #27 dopo CI sintetica superata; HW-MANUAL-001 e' stata
  accettata personalmente il 16 ottobre 2026 sulla revisione
  `59b26fd6b427e6756ce0285fee60590cb077ebf1`, con sequenza PY -> DEP -> DIAG -> HW.
  GPU e compatibilita' di inferenza non sono state verificate
  ne' autorizzate in HW-SCOPE-001.

## GPU-SCOPE-001: provider NVIDIA read-only prima di AMD e Intel

- Stato: accepted; approvazione esplicita dell'utente il 16 ottobre 2026.
- Funzioni: `hardware.gpu.nvidia.detect`, `hardware.gpu.detect`;
  modello, conteggio, VRAM totale e stato di interrogazione, con
  distinzione tra utility mancante, driver non interrogabile, timeout
  e risposte vuote o parziali. Nessun dato GPU salvato su Git.
- Ingressi: esclusivamente nvidia-smi presente, subprocess senza shell
  e con timeout, parametri fissi e parsing bounded. Non installare nulla.
- Ingresso Ruth: quinto passaggio di `main.py`, dopo HW CPU/RAM,
  indipendente da un virtualenv conforme o dalla presenza della utility.
- Non approvate: migrazioni AMD/Intel, driver, CUDA e backend,
  benchmark, memoria disponibile, inferenza e diagnostica nativa.
- Controlli: test sintetici, CI e mappe prima di ogni promozione;
  GPU-MANUAL-001 personale dopo merge, prima di altra migrazione.
- Vincoli: MAIN-FLOW-001, MIG-FLOW-001/002, REMOTE-001, DATA-001.

- Applicazione GPU-SCOPE-001 (16 ottobre 2026): codice e test
  sintetici verificati su GitHub e integrati sino a main tramite PR #33.
  GPU-MANUAL-001 completata positivamente il 16 ottobre 2026 sul
  commit `c344c71013cfa05da4f49f3204900f4bdc89b8ea`, con quinta fase NVIDIA `available` e
  sequenza Ruth conclusa. AMD, Intel, driver, CUDA e backend restano
  fuori dall'ambito e non sono stati verificati o autorizzati.

## CAT-SCOPE-001: catalogo locale dei modelli dichiarati

- Stato: accepted, approvazione esplicita dell'utente 16 ottobre 2026.
- Manifest opzionale JSON sotto `.local/models/catalog.json`, soltanto
  file esplicitamente elencati, riferimenti relativi confinati alla
  cartella locale, senza percorso assoluto, traversal o symlink.
- API: `models.catalog.catalog` e `resolve`, report immutabili;
  riconoscere non configurato, vuoto, pronto, parziale e non valido.
  `present_unverified` non certifica il formato o inferenza.
- MAIN-FLOW-001: sesta fase automatica dopo NVIDIA, indipendente da
  altre fasi; controllo mirato `main.py catalog`. Un catalogo assente
  o vuoto non e' errore; dati invalidi o file dichiarati mancanti
  vengono segnalati, senza modifiche all'ambiente.
- Proibiti scanning generale del disco, download, verifica binaria GGUF,
  acquisizione Hugging Face, runtime inference e pubblicazione manifest.
- Test sintetici, CI, branch fino a main, poi CAT-MANUAL-001 obbligatorio
  prima di un nuovo componente (MIG-FLOW-001/002, DATA-001, REMOTE-001).

- Applicazione CAT-SCOPE-001 (16 ottobre 2026): implementazione
  verificata in CI e integrata in main con PR #38.
  CAT-MANUAL-001 completato positivamente dall'utente il 16 ottobre 2026
  sul commit `90e57f106fb664b0257775260044125ea85228e2` (MIG-FLOW-002); GGUF,
  Hugging Face, scansione, acquisizione e inferenza non inclusi.
  Nessuna nuova porzione funzionale e' approvata implicitamente.

## GGUF-SCOPE-001: intestazione dei soli GGUF catalogati

- Stato: accepted; approvazione esplicita dell'utente il 16 ottobre 2026.
- Scopo: esaminare i primi 24 byte dei soli artefatti GGUF dichiarati da
  CAT-001; leggere firma, versione e contatori, restituendo report immutabili
  e codici di errore portabili, senza certificare il file intero.
- Ingresso: settimo passaggio automatico di `main.py`, dopo CAT, con
  skip non bloccante per catalogo assente/vuoto; comando mirato GGUF.
- Proibiti: scansione file non catalogati, download, rete, installazioni,
  lettura metadata/tensori/pesi, Hugging Face, CUDA e inferenza.
- Verifica: fixture sintetiche, CI, mappe e PR fino a main; successivo
  GGUF-MANUAL-001 personale obbligatorio secondo MIG-FLOW-002.
- Vincoli: MIG-FLOW-001/002, REMOTE-001, DATA-001 e MAIN-FLOW-001.

### Applicazione GGUF-SCOPE-001 (16 ottobre 2026)

- Implementazione e verifica CI integrate in main tramite PR #41 (leaf),
  #42 (formats), #43 (Models), #44 (root), merge `15b106082c4dee37622c61a8afefd248a57464a0`.
- CI success in tutti e quattro i passaggi (runs 38935705385,
  38935861881, 38935943414, 38936032188), compresi test GGUF
  sintetici, test di ingresso e verifica mappe.
- GGUF-MANUAL-001 ancora pending fino a prova personale dell'utente
  dalla revisione main sincronizzata; nessun componente successivo e'
  automaticamente autorizzato.

### GGUF-MANUAL-001: collaudo positivo (16 ottobre 2026)

- Stato: complete/passed, conferma esplicita dell'utente.
- Revisione eseguita personalmente con Ruth senza argomenti:
  `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9` (main).
- Sette controlli in sequenza, catalogo/GGUF `not_configured`,
  completamento regolare. Nessun file GGUF reale ispezionato:
  il collaudo non certifica integrita' o capacita' di inferenza.
- Il perimetro GGUF-SCOPE-001 resta invariato. Nessuna nuova
  migrazione autorizzata implicitamente; applicare MIG-FLOW-001/002.

## GGUF-META-SCOPE-001: metadati tecnici selezionati

- Stato: accepted; approvazione esplicita il 16 ottobre 2026.
- Ingressi: solo modelli GGUF presenti nel catalogo CAT-001.
- Uscite: API `inspect_metadata` e `inspect_all_metadata`, report
  immutabili e campi opzionali dichiarati architecture, file_type
  e context_length; nessuna compatibilita' inferenza certificata.
- Limiti bounded a metadati, stringhe, numero di voci, array e
  profondita'; codici portabili su errori e dati malformati.
- Ottavo step automatico dopo GGUF, sottocomando `gguf-metadata`;
  catalogo assente = skip valido. Nessuna scrittura/download/rete.
- Esclusi pesi, tensor payload, backend, CUDA, Hugging Face e
  inferenza. Migrazione GitHub-first con CI e maps, poi collaudo
  personale GGUF-META-MANUAL-001 obbligatorio (MIG-FLOW-002).
- Nessuna porzione successiva approvata implicitamente.

### Applicazione GGUF-META-SCOPE-001 (16 ottobre 2026)

- Codice implementato e verificato con CI sintetica; promozioni
  PR #47 (GGUF), #48 (Formats), #49 (Models) e #50 (main).
- CI success: runs 38940033886, 38940265778, 38940360999,
  38940625555. Il job finale e' stato riattivato senza
  modifiche al contenuto candidato, poi ha restituito success.
- Commit di integrazione main: `a630c83f6c62a229e18fe5e64f0863c19de16018`.
- GGUF-META-MANUAL-001 **pending**: ancora necessaria conferma
  personale esplicita dell'utente dopo avvio di Ruth (MIG-FLOW-002).
- Il perimetro resta bounded/read-only; nessun tensor, download,
  compatibilita' runtime o inferenza sono stati approvati.

### GGUF-META-MANUAL-001: accettazione positiva (16 ottobre 2026)

- Stato: **complete/passed** a seguito di conferma esplicita dell'utente.
- Collaudo effettuato sul commit `d318fd793da514f21ee2e8475304d1a6c883ee94`, con avvio di
  `main.py` senza argomenti e otto controlli nell'ordine previsto.
- Il caso catalogo assente e' valido: GGUF e metadati
  `not_configured` sono stati gestiti senza errore.
- Nessun file GGUF reale ispezionato personalmente; l'accettazione
  non dimostra integrita' dei modelli o compatibilita' inferenza.
- GGUF-META-SCOPE-001 resta invariato. Nessuna nuova migrazione
  approvata implicitamente (MIG-FLOW-001/002 e DATA-001).

## GGUF-INDEX-SCOPE-001 — verifica strutturale dichiarativa

- Stato: accepted; via libera esplicito dell'utente il 16 ottobre 2026.
- Soli GGUF dichiarati in CAT, API inspect_tensor_index e
  inspect_all_tensor_indexes, report immutabili con soli codici e conteggi.
- Indici: numero nomi e unicita', massimo 4 dimensioni positive,
  limiti aritmetici, offset allineati e dentro la dimensione del file;
  solo tipi scalari con dimensione conosciuta verificano gli intervalli.
- I tipi quantizzati/sconosciuti restano non verificabili per ingombro:
  stato size_unverified, non una verifica positiva dell'intero modello.
- No peso, payload, hash, runtime o inferenza. Solo metadata
  bounded, con limitazioni e difese CAT/GGUF preesistenti.
- Nuova nona fase Ruth dopo META, sottocomando gguf-index;
  catalogo assente = skip regolare. Fixture sintetiche, CI,
  mappe e PR lungo tutti gli antenati. Collaudo personale
  GGUF-INDEX-MANUAL-001 post-merge (MIG-FLOW-002).
- DATA-001, REMOTE-001, MAIN-FLOW-001 restano vigenti.

### GGUF-INDEX-001: integrazione verificata (16 ottobre 2026)

- Approvazione GGUF-INDEX-SCOPE-001 immutata. PR #53-#56 integrate
  fino a main `1d51cfeed2bec83119ebf56ff0cb3840089484cf`; CI success runs 38945221302, 38945375637,
  38945463381, 38945568401; test sintetici e mappe superati.
- Il validator controlla solo struttura dei descriptor e offset,
  non legge tensor data e non stima ingombri quantizzati senza
  regole sufficienti: `size_unverified` e' risultato non confermato.
- GGUF-INDEX-MANUAL-001 **pending**; il collaudo dell'utente richiede
  Ruth senza argomenti dalla nuova revisione main.
- Prossimo perimetro non autorizzato prima di conferma positiva
  del test personale (MIG-FLOW-002).

### GGUF-INDEX-MANUAL-001: accettazione personale (17 ottobre 2026)

- GGUF-INDEX-MANUAL-001: **complete/passed**, accettazione esplicita
  dell'utente il 17 ottobre 2026 dopo esecuzione personale del flusso
  ordinario `.venv/bin/python main.py` sul commit `09045243e944514d40b9e58c6159b013efa2a196`,
  identico a main verificato su GitHub.
- Nove fasi in ordine PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT ->
  GGUF -> META -> INDEX; riepilogo finale di completamento regolare.
- Catalogo e letture GGUF/META/INDEX `not_configured` sono skip validi
  e non bloccanti. Nessun file GGUF reale ispezionato personalmente;
  non risultano validati contenuti dei tensori, pesi o compatibilita'
  con CUDA/backend/inferenza.
- La prova manuale completa il gate MIG-FLOW-002 per MIG-009;
  una nuova migrazione richiede uno scope separato e approvazione
  esplicita (MIG-FLOW-001). Nessun nuovo componente avviato.
- DATA-001 e REMOTE-001 invariati: in Git solo esito e commit;
  niente path, nomi modelli, output grezzi, inventari, log o benchmark.
GGUF-INDEX-SCOPE-001 resta invariato; nessuna migrazione successiva
autorizzata implicitamente.

## GGUF-BLOCK-SCOPE-001: blocchi quantizzati dichiarativi (17 ottobre 2026)

- Stato: **approved** per MIG-010; base main `6f472b0ac182b113c5c727c3f8f5c291c074de2e`.
- Cinque tipi GGML: Q4_0 (id 2, 32:18), Q4_1 (3, 32:20),
  Q5_0 (6, 32:22), Q5_1 (7, 32:24), Q8_0 (8, 32:34).
  Primo asse divisibile per 32; estensione in byte dal numero
  blocchi. Altri tipi restano `size_unverified`.
- Solo modelli GGUF nel catalogo locale CAT-001, reuse parser
  e difese INDEX, limiti su strutture, nomi e contatori.
  Controlli offset, sovrapposizioni ed estremi del file.
- API immutabili inspect_block_layout / inspect_all_block_layouts;
  decimo passaggio `BLOCKS` automatico dopo `INDEX`,
  `main.py gguf-blocks` mirato. Catalogo assente: skip valido.
- Non si leggono o dequantizzano pesi. Nessun download,
  rete, inferenza, CUDA, backend, benchmark o dati macchina.
- Fixture sintetiche, CI, mappe e PR fino a main; poi collaudo
  personale GGUF-BLOCK-MANUAL-001 obbligatorio. Nessuna
  migrazione seguente implicitamente approvata.
- Si applicano MAIN-FLOW-001, MIG-FLOW-001/002, DATA-001, REMOTE-001.

### GGUF-BLOCK-001: integrazione verificata (17 ottobre 2026)

- Scope approvato GGUF-BLOCK-SCOPE-001 invariato.
- PR #59 (leaf), #60 (Formats), #61 (Models), #62 (root)
  tutte integrate, CI runs 39022476943, 39022574327,
  39022637716, 39022698747: **success**, con test sintetici
  GGUF, regressioni CLI e controllo delle mappe.
- Integrazione main `5f896f1bbacaa8c802f66f2a40ee7663cfa40604`. Supportati unicamente i cinque
  GGML type IDs 2/3/6/7/8 con formula blocchi.
  Altri tipi `size_unverified`; nessun dato dei pesi letto.
- GGUF-BLOCK-MANUAL-001 ancora **pending** (MIG-FLOW-002).
  Nessuna nuova migrazione autorizzata implicitamente.

### GGUF-BLOCK-MANUAL-001: accettazione positiva (17 ottobre 2026)

- L'utente ha eseguito personalmente il flusso di dieci controlli
  di `.venv/bin/python main.py` sulla revisione `2f42a9d7a631e8779613784b58f8ce1b92afff9c`,
  coincidente con il main pubblicato; ha confermato esplicitamente
  positivo il collaudo GGUF-BLOCK-MANUAL-001 il 17 ottobre 2026.
- Tutte le fasi PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF ->
  META -> INDEX -> BLOCKS sono state completate in ordine.
  Il catalogo opzionale assente e' risultato `not_configured`
  e GGUF/META/INDEX/BLOCKS sono skip validi e non bloccanti.
- Non e' stato collaudato personalmente alcun file GGUF reale:
  l'esito non certifica pesi, integrita', dequantizzazione,
  CUDA, backend, runtime o inferenza.
- I risultati di CI e la verifica con fixture sintetiche dei cinque
  tipi Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 restano distinti
  dall'accettazione manuale senza catalogo.
- DATA-001 e REMOTE-001 rispettati: non vengono versionati
  inventari dell'host, percorsi reali, log, manifest, modelli,
  benchmark o credenziali. Nessuna nuova migrazione approvata.
- Stato: **complete/passed** per il gate manuale di MIG-010
  ai sensi di MIG-FLOW-002, senza estendere GGUF-BLOCK-SCOPE-001.
- L'implementazione era stata verificata in CI e integrata con
  PR #59-#62; ora anche il collaudo personale e' stato accettato.
  Migrazione successiva soltanto dopo specifica approvazione.

## GGUF-KBLOCK-SCOPE-001 — sei superblocchi K (17 ottobre 2026)

- Approvazione esplicita dell'utente, base main `1b497172bfad562e6096e88ac8be69ef01a8ced6`; MIG-011.
- Sei GGML ID: 10 Q2_K (256:84), 11 Q3_K (256:110),
  12 Q4_K (256:144), 13 Q5_K (256:176), 14 Q6_K (256:210),
  15 Q8_K (256:292), secondo ggml.h/ggml-common.h.
- Calcolo ingombri di soli descrittori GGUF in CAT con la
  prima dimensione divisibile per 256; controlli bounded,
  allineamento, limiti file e sovrapposizioni. Altri
  tipi restano `size_unverified`.
- Nessuna modifica implicita alle precedenti API INDEX/BLOCKS,
  che continueranno a non attestare tipi K.
- API inspect_kblock_layout / inspect_all_kblock_layouts
  immutabili. Undicesimo passaggio `KBLOCKS`, comando
  mirato `gguf-kblocks`, catalogo assente = skip valido.
- Fixture sintetiche anche per successivo collaudo manuale
  personale; test/CI/mappe e promozioni fino a main.
  GGUF-KBLOCK-MANUAL-001 post-merge obbligatorio.
- Non leggere tensor payload o dequantizzare, no rete,
  download, CUDA, benchmark, installazioni, inferenza.
  DATA-001, REMOTE-001, MIG-FLOW-001/002 invariati.

### GGUF-KBLOCK-001: integrazione verificata (17 ottobre 2026)

- Approvazione GGUF-KBLOCK-SCOPE-001 invariata.
- PR #65 leaf, #66 Formats, #67 Models, #68 main,
  CI success runs 39025028855, 39025109274, 39025159357,
  39025210189 (fixture, regressioni, mappe); root main
  `a803e15e872d71a96b7b2de37b13a8964d37d168`.
- Sei tipi GGML 10-15 con 256 elementi per blocco; byte
  84/110/144/176/210/292 e prima dimensione divisibile per 256.
  Solo descrittori, mai lettura/dequantizzazione pesi.
- API INDEX e BLOCKS mantengono semantica precedente, tipi K
  `size_unverified` prima della nuova fase KBLOCKS.
- GGUF-KBLOCK-MANUAL-001 **pending**, test personale Ruth
  undici fasi e fixture sintetiche locale post-merge, poi
  accettazione esplicita richiesta. Nuovo scope non approvato.

## MIG-FLOW-002-R2: accettazione automatica da evidenza positiva (17 ottobre 2026)

- Stato: accepted, disposizione esplicita dell'utente.
- La prova personale incollata con SHA main corrispondente, fasi concluse
  e test prescritti superati vale come accettazione, senza secondo consenso.
- Dopo verifica completare autonomamente la chiusura GitHub e proporre
  lo scope della migrazione successiva; MIG-FLOW-001 continua a richiedere
  approvazione esplicita **dello scope nuovo**, prima di modificare codice.
- In caso di prova ambigua, errore o SHA incongruente non dichiarare
  superato il collaudo. Non confondere skip valido `not_configured`
  con verifica su GGUF reale. I risultati reali rimangono locali
  (DATA-001); solo SHA, stato e conteggi generici possono essere registrati.
- Sostituisce l'obbligo di doppia approvazione dopo log positivo
  previsto dalla vecchia formulazione di MIG-FLOW-002.
