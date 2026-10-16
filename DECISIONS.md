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
- Ambito corrente: PY-001 e' gia in environment/python/main; descrizione
  PY-SCOPE-001 sottoposta all'utente prima delle promozioni restanti.
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

- Stato: accepted.
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
