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
- Decisione: ogni porzione integrata in main deve essere eseguita dal
  punto d'ingresso ufficiale di Ruth e produrre messaggi a terminale relativi
  alla funzione e al suo esito. Dopo l'integrazione l'utente avvia Ruth
  localmente di persona; la migrazione successiva attende la sua conferma
  esplicita dell'esito positivo.
- Verifica: fornire comando ed esiti attesi e registrare nel checkpoint la
  revisione provata e l'accettazione ancora in attesa oppure confermata.
  Test automatici o esecuzioni dell'agente non sostituiscono il collaudo.
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
