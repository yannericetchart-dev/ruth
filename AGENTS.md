# Ruth: ingresso per gli agenti

Il repository GitHub `yannericetchart-dev/ruth` e' la fonte comune del progetto.
All'inizio di una nuova attivita' leggere `START_HERE.md` dal branch `main`
su GitHub; fissare il commit letto e seguire i rimandi alle decisioni, al
checkpoint e alle mappe. Confrontare la revisione pubblicata con il checkout
locale prima di modificare file. Dichiarare quando GitHub non e' accessibile.

Prima di sviluppare riportare brevemente: obiettivo, attivita', decisioni
vigenti, branch/revisione, limiti, problemi aperti e prossima azione. Non
interpretare una proposta come decisione approvata o uno skeleton come codice
funzionante. Una modifica locale non e' pubblicata finche' non e' su GitHub.

## Continuita' CONT-001

- Leggere le istruzioni degli antenati e le mappe del componente interessato.
- Aggiornare decisioni, checkpoint e mappe dopo ogni passaggio significativo.
- Un cambio di strategia deve indicare la decisione sostituita, la motivazione
  e le attivita' coinvolte prima di ulteriore sviluppo dipendente.
- Propagare gli effetti fino alla radice; distinguere sviluppo, verifica e
  integrazione. Ogni attivita' concorrente deve avere un checkpoint proprio.
- In caso di documenti contraddittori, controllare le fonti e dichiarare la
  divergenza prima di effettuare modifiche che dipendono dalla decisione.

## Portabilita' DATA-001

Versionare solo codice, documentazione di progetto, schemi, configurazioni
generiche e fixture sintetiche. Nessun dato reale dipendente da host o utente:
percorsi assoluti personali, credenziali, UUID, PID reali, inventari hardware,
log, risultati di benchmark, cache, modelli scaricati e build restano locali.
Usare `.local/` per risultati e stato della macchina; la cartella e' ignorata.
Le evidenze locali sono richiamate soltanto con identificativi portabili.
Le decisioni tecniche sono documentabili senza copiare telemetria o log.
Una policy per un'architettura supportata e' generica; la rilevazione della
macchina specifica non lo e'. Non copiare lock o profili costruiti per un host
come se fossero requisiti universali. Non riscrivere la storia dei branch
esistenti per rimuovere artefatti senza un piano separato.

## Sequenza autorizzata

1. Configurare e verificare la ripresa in Codex e ChatGPT.
2. Pubblicare lo scaffold e le mappe senza migrare codice applicativo.
3. Migrare un componente alla volta con i test pertinenti.

Non anticipare la migrazione finche' i precedenti passaggi non sono verificati.
Kaggle e' inattivo e fuori dal perimetro di sviluppo corrente. Il codice
storico deve essere censito senza attivarlo o installarne le dipendenze.
Lo stato corrente del processo e' indicato in `CHECKPOINT.md`.
