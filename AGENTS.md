# Ruth: ingresso per gli agenti

Il repository GitHub `yannericetchart-dev/ruth` e' la fonte comune del progetto.
All'inizio di una nuova attivita' leggere `START_HERE.md` dal branch `main`
su GitHub; fissare il commit letto e seguire i rimandi alle decisioni, al
checkpoint e alle mappe. Il checkout locale, se accessibile, si puo' leggere
per diagnosi ma non va modificato durante lo sviluppo: REMOTE-001 impone
pubblicazione direttamente su GitHub e sincronizzazione locale soltanto dopo
il merge in main. Dichiarare quando GitHub non e' accessibile.

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
funzionante. Nessuna modifica locale al checkout principale e' fonte di
sviluppo: ogni modifica versionata deve essere pubblicata direttamente su
GitHub e identificata con commit o PR.

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


## Sviluppo remoto e sincronizzazione finale (REMOTE-001)

Decisione vigente: REMOTE-001 sostituisce la precedente SYNC-001.
**ChatGPT e Codex seguono lo stesso flusso GitHub-first**: per ogni
modifica versionata leggere il commit main attuale, usare i branch di lavoro
e i commit su GitHub, aprire le PR, svolgere revisioni, aggiornare mappe e
checkpoint e promuovere fino a main entro il perimetro approvato. Non
scrivere, creare, modificare, cancellare, formattare, generare o mettere
in stage file nel checkout principale del progetto, neppure se Codex ha
accesso alla cartella. Non eseguire checkout, merge, pull o reset del
clone principale durante sviluppo. Non usare i cambiamenti non committati
del clone come fonte canonica: non importarli, pubblicarli o cancellarli
senza una decisione separata. In caso di assenza di accesso GitHub o di
strumenti di scrittura remota, dichiarare il blocco: non ripiegare
silenziosamente sul checkout principale.

**Prove locali permesse solo in spazi isolati ed esclusi dalla sync Git**:
directory temporanee fuori dal checkout o sotto `.local/verification/`
(ignorata da Git), usando snapshot puliti del *commit candidato pubblicato
su GitHub*. Test, fixture, generazione delle mappe e build temporanee
possono modificare esclusivamente queste copie e le directory di cache
escluse, mai il checkout principale. Eseguire le verifiche sullo stesso
contenuto candidato alla promozione; ogni correzione destinata al progetto
deve essere ripubblicata con gli strumenti GitHub e verificata nuovamente.
Non includere cache, dati, log o misure locali nei commit (DATA-001).
Non dichiarare superati test che non sono stati eseguiti.

**Sincronizzazione solo al termine dell'integrazione in main**: prima
completare autonomamente il flusso MIG-FLOW-001 fino al merge e registrare
lo SHA di main. Poi, e soltanto allora, confrontare il clone principale
locale con quello SHA e, se si dispone di accesso autorizzato, aggiornarlo
in modo non distruttivo. In caso di assenza di accesso locale, dare
all'utente le istruzioni di PROJECT_SETUP.md e registrare
`local_sync: pending_no_access` senza bloccare retroattivamente la fase
di sviluppo remoto. Se il checkout e' sporco, su altro branch o divergente,
non forzare reset, clean, stash, rebase, scarti, cambio branch o merge:
segnalare `local_sync: blocked_divergence` finche' venga scelta una
procedura sicura. Nessuna modifica locale preesistente deve essere persa.

Separare sempre: codice pubblicato e verificato su GitHub, integrazione in
main, sincronizzazione del checkout personale e collaudo manuale
MIG-FLOW-002. La migrazione successiva resta bloccata finche' l'utente
non conferma il collaudo locale positivo; non e' bloccata la promozione
gia' autorizzata dalla mera assenza di sincronizzazione preliminare.
In ChatGPT Ruth si usa Chat + GitHub, non Work come soluzione implicita
per accedere al filesystem locale.


## Mappe ricorsive

Seguire ARCHITECTURE.md. Modificare NODE.yaml (sottoinsieme JSON di YAML),
poi rigenerare con `python3 tools/maps.py --write` e verificare con
`python3 tools/maps.py --check`. Non modificare soltanto una vista Markdown.
La rigenerazione propaga i riepiloghi; aggiornare anche semanticamente le
attivita bloccate, le decisioni e i checkpoint degli antenati interessati.
Nessun risultato locale deve essere copiato nelle mappe.
