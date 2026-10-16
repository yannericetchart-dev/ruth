# Ruth: ingresso per gli agenti

Il repository GitHub `yannericetchart-dev/ruth` e' la fonte comune del progetto.
All'inizio di una nuova attivita' leggere `START_HERE.md` dal branch `main`
su GitHub; fissare il commit letto e seguire i rimandi alle decisioni, al
checkpoint e alle mappe. Confrontare la revisione pubblicata con il checkout
locale prima di modificare file. Dichiarare quando GitHub non e' accessibile.

Per fissare la revisione, leggere prima il ref `heads/main` tramite il
connettore GitHub: `https://api.github.com/repos/yannericetchart-dev/ruth/git/ref/heads/main`.
Usare `object.sha` come commit e leggere START_HERE.md e i documenti
successivi con quel SHA nel parametro ref. In alternativa leggere la branch
API e usare `commit.sha`. Lo SHA del blob restituito da un file non identifica
il commit del repository. Se il ref non e' risolvibile, dichiarare la ripresa
incompleta e non iniziare sviluppo dipendente da una revisione presunta.

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

## Migrazioni: semaforo verde sul perimetro (MIG-FLOW-001)

Prima di ogni porzione presentare all'utente una descrizione delle funzioni:
scopo, ingressi, risultati, controlli, effetti sui dati, limiti e verifiche.
Attendere il suo semaforo verde su quel perimetro prima di migrare.
Registrare perimetro e approvazione nel checkpoint, senza dedurli dal silenzio.

Dopo il verde, completare autonomamente implementazione, correzioni, test,
revisioni, PR e promozioni lungo tutti i branch padri fino all'integrazione
in main, aggiornando mappe e checkpoint. Non richiedere ulteriori conferme
per ciascuna promozione gia compresa nel perimetro approvato. Fermarsi per
dubbi o domande che richiedono una decisione dell'utente, inclusa una modifica
sostanziale del perimetro. Una normale scelta tecnica interna non e' un nuovo
passaggio di approvazione. Non iniziare la porzione seguente senza averne
prima presentato le funzioni e ricevuto il relativo verde.

## Mappe ricorsive

Seguire ARCHITECTURE.md. Modificare NODE.yaml (sottoinsieme JSON di YAML),
poi rigenerare con `python3 tools/maps.py --write` e verificare con
`python3 tools/maps.py --check`. Non modificare soltanto una vista Markdown.
La rigenerazione propaga i riepiloghi; aggiornare anche semanticamente le
attivita bloccate, le decisioni e i checkpoint degli antenati interessati.
Nessun risultato locale deve essere copiato nelle mappe.
