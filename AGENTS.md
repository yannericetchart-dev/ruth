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

## Migrazioni: ingresso reale e collaudo manuale (MIG-FLOW-002)

Ogni porzione migrata deve essere eseguita attraverso il punto d'ingresso
ufficiale di Ruth in main, con messaggi a terminale corrispondenti alla
funzione e al suo esito. Un import, una API isolata o i soli test non
soddisfano questo requisito. Includere questo percorso nella descrizione
funzionale sottoposta all'utente prima del verde.

Dopo l'integrazione in main fornire all'utente il comando di avvio e i
messaggi/esiti attesi. L'utente deve avviare Ruth localmente di persona e
confermare esplicitamente l'esito positivo. Registrare nel checkpoint la
revisione di main provata e lo stato del collaudo; log e risultati reali
rimangono locali secondo DATA-001. I test automatici e le esecuzioni
dell'agente non sostituiscono questa conferma, ne' il silenzio la implica.

Non iniziare la migrazione successiva prima della conferma positiva
dell'utente. Distinguere integrazione tecnica completata da accettazione
manuale ancora in attesa. Se il collaudo fallisce, restare sulla porzione
corrente, correggerla e sottoporla nuovamente al collaudo dell'utente.


## Sincronizzazione e blocco di ripresa (SYNC-001)

GitHub e checkout locale sono stati distinti. Prima di riprendere un lavoro
interrotto, di proseguire un'attivita' dipendente dal checkout o di dichiarare
completata una migrazione, verificare lo SHA di main su GitHub e lo stato del
checkout locale effettivo. Controllare remote, branch, HEAD, origin/main e
modifiche locali (anche file non tracciati). Non dedurre l'allineamento locale
dalla disponibilita' della connessione GitHub.

Se si ha accesso autorizzato al checkout, sincronizzarlo solo quando e'
sicuro e in fast-forward, conservando le modifiche dell'utente, quindi
ricontrollare SHA e stato. Mai eseguire automaticamente reset --hard, clean,
force-push, stash, rebase, scarti o cambi branch potenzialmente distruttivi.

Se **non** si ha accesso al checkout (ad esempio Chat senza accesso alla
cartella), fermarsi al punto di sincronizzazione: comunicare subito all'utente
lo SHA target di main e i comandi concreti, non distruttivi, per verificare e
sincronizzare manualmente il clone. Fornire queste istruzioni **prima di
riprendere e terminare il lavoro dipendente dallo stato locale**, non solo
alla consegna finale. Chiedere il riscontro dell'esito o della divergenza,
senza domandare nuovamente un'autorizzazione funzionale gia' concessa.
Se la verifica o la sincronizzazione non e' completata, mantenere
`local_sync: pending_no_access` (oppure `blocked_divergence`) nel
checkpoint e dichiarare la ripresa/consegna incompleta: nessuna presunzione
di locale aggiornato o di collaudo personale.

Il blocco di sincronizzazione non revoca il verde MIG-FLOW-001: dopo il
riscontro si riprende e si completa autonomamente quanto autorizzato fino
a main, interrompendosi nuovamente solo per problemi reali. La conferma
manuale MIG-FLOW-002 e' comunque distinta e obbligatoria per abilitare
la migrazione seguente.

Per il workflow ChatGPT di Ruth, utilizzare Chat e il connettore GitHub
senza passare a Work per colmare la mancanza di accesso locale: Work e
Codex possono consumare una quota condivisa che l'utente vuole preservare.
Le istruzioni di sincronizzazione vanno date per l'esecuzione personale.
Seguire PROJECT_SETUP.md. Non registrare in Git percorsi personali, log,
dati di host/utente o risultati reali, conformemente a DATA-001.

## Mappe ricorsive

Seguire ARCHITECTURE.md. Modificare NODE.yaml (sottoinsieme JSON di YAML),
poi rigenerare con `python3 tools/maps.py --write` e verificare con
`python3 tools/maps.py --check`. Non modificare soltanto una vista Markdown.
La rigenerazione propaga i riepiloghi; aggiornare anche semanticamente le
attivita bloccate, le decisioni e i checkpoint degli antenati interessati.
Nessun risultato locale deve essere copiato nelle mappe.
