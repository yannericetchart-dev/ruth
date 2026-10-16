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
