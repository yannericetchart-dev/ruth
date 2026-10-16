# Ruth: configurazione e collaudo della continuita'

Protocollo: CONT-001, revisione 1. Fonte comune: GitHub, branch `main`.
Questo documento e' una procedura generica, senza percorsi o account locali.

## Codex (REMOTE-001)

Il clone principale e' di sola lettura durante lo sviluppo. Codex deve
risolvere main su GitHub e leggere START_HERE.md, AGENTS.md, DECISIONS.md
e CHECKPOINT.md a quel commit, anche se i corrispondenti file locali non
sono aggiornati. Non usare un AGENTS.md locale datato per sostituire le
istruzioni vigenti su GitHub. Il lavoro versionato avviene tramite GitHub:
branch di lavoro remoto, commit e PR, revisioni e merge fino a main.
Non scrivere file tracciati o nuovi file destinati al repository nel
checkout principale, non eseguire autoformat o generatori su di esso,
non lanciare comandi che modifichino il suo stato Git.
Le modifiche locali preesistenti restano intatte e non sono parte del lavoro.

Quando servono test, verifiche delle mappe, build o comandi da terminale,
materializzare **uno snapshot della revisione GitHub candidata** in una
directory temporanea esterna oppure in `.local/verification/` ignorata.
Eseguire i test interamente da questa copia, usare cache e output temporanei
al suo interno e non riportare nel clone principale le modifiche generate.
Un modo possibile, quando il commit candidato e' disponibile nell'object
store, e' `git archive <commit>` estratto in tale directory; se manca,
scaricare il commit con un fetch che non cambi branch/working tree e poi
materializzarlo. Non usare una worktree come scorciatoia per scrivere
nel checkout principale. Pubblicare ogni correzione versionata su GitHub
e verificare di nuovo la revisione risultante.

Solo **dopo** l'integrazione in main procedere alla sincronizzazione
sicura del checkout locale secondo la procedura finale sotto. Se il clone
non e' pulito o e' su un branch diverso, non sovrascriverlo né forzare:
segnalare il blocco e chiedere come preservare i dati.

## ChatGPT: istruzioni del progetto

Conservare le istruzioni precedenti compatibili. Aggiungere questo blocco
alle istruzioni del Progetto associato a Ruth, senza inserirvi dettagli di
macchina. Non e' sufficiente allegare il documento a una singola chat.

```text
Progetto: yannericetchart-dev/ruth. Protocollo CONT-001 revisione 1.
All'inizio di ogni nuova chat o ripresa del lavoro, usa la connessione GitHub
per risolvere prima il commit main tramite
https://api.github.com/repos/yannericetchart-dev/ruth/git/ref/heads/main
(campo object.sha; in alternativa branch API, campo commit.sha).
Leggi poi START_HERE.md e gli altri documenti usando quel commit come ref.
Lo SHA del blob di un file non e' lo SHA del commit. Se non riesci a risolvere
il ref, dichiara la ripresa incompleta prima di sviluppare.
Segui i riferimenti a AGENTS.md, DECISIONS.md, CHECKPOINT.md e alle
mappe applicabili. Non presumere che un file caricato o una chat precedente
rappresenti l'ultima revisione. Se GitHub non e' accessibile, dichiaralo.
Usa GitHub come unica fonte di modifiche versionate. Non modificare il
checkout principale locale durante lo sviluppo; per test usa solo copie
isolate ignorate da Git o cartelle temporanee esterne. Sincronizza il clone
locale solo dopo il merge definitivo in main secondo REMOTE-001.
Prima di modificare codice, riassumi obiettivo, fase, decisioni, limiti e
prossima azione. Distingui pianificato, implementato, verificato e integrato.
Aggiorna checkpoint, decisioni e mappe entro il perimetro autorizzato. Se non
puoi pubblicare lo stato aggiornato, dichiara il passaggio di consegne incompleto.
DATA-001: log, risultati benchmark, credenziali e dati reali di host/utente
rimangono locali ed esclusi da Git. Kaggle e' inattivo.
```

## Collaudo indipendente

Eseguire in una nuova chat/sessione per ciascun client, senza incollare la
cronologia precedente. Messaggio di prova: `Riprendi il progetto Ruth`.
Non introdurre modifiche di codice durante il collaudo.

Esito positivo soltanto se il client:

1. legge il punto di ingresso da GitHub e identifica la revisione;
2. ricostruisce fase e prossima azione dal checkpoint corrente;
3. riconosce Kaggle inattivo e il vincolo sui dati locali;
4. distingue configurazione, scaffold e migrazione applicativa;
5. dichiara fonti mancanti o divergenze anziche' presumere esiti positivi.

Registrare transcript, nomi dei client e risultati in `.local/verification/`.
In Git riportare soltanto lo stato generale del processo, senza log o dati
personali. Aggiornare CHECKPOINT.md dopo la verifica dei due client.

## Sincronizzazione

Le istruzioni del Progetto ChatGPT e AGENTS.md sono due ingressi distinti:
modificare il file locale non aggiorna automaticamente le impostazioni della
UI. Dopo una modifica del protocollo verificarli entrambi. Commit e push
rendono condivise le modifiche, mentre un commit soltanto locale non lo fa.

### Sincronizzazione finale, solo dopo il merge in main (REMOTE-001)

Lo sviluppo e i test devono gia' essere completati su GitHub prima di
sincronizzare il clone principale. La sincronizzazione locale non e'
un prerequisito per riprendere il lavoro remoto, contrariamente alla
vecchia SYNC-001. ChatGPT Ruth usa Chat + GitHub (non Work).

Al termine delle promozioni leggere di nuovo il ref heads/main via GitHub
e comunicare all'utente il commit target. Se il client ha accesso al
clone principale, rilevarne **senza scriverlo** lo stato; altrimenti
dare all'utente i seguenti comandi per la verifica, da eseguire nella
cartella locale di Ruth:

```sh
git remote -v
git status --short --branch
git branch --show-current
git rev-parse HEAD
git fetch origin
git rev-parse origin/main
git status --porcelain --untracked-files=all
```

Se origin non punta al repository canonico, se il checkout e' sporco,
se il branch attivo non e' main o se origin/main non coincide con lo SHA
GitHub comunicato, **fermarsi senza alterare nulla** e segnalare il
problema. Su un clone pulito, branch main, remote corretto e aggiornamento
fast-forward possibile, completare con:

```sh
git merge --ff-only origin/main
git rev-parse HEAD
git status --short --branch
```

Confrontare il nuovo HEAD con il commit GitHub target e registrare
l'esito nel checkpoint: `local_sync: verified`, `pending_no_access`
oppure `blocked_divergence`. Non usare reset --hard, clean, stash,
rebase, cambi branch automatici o force-push per nascondere lavoro locale.
In caso di blocco, non cancellare ne' pubblicare automaticamente i file
esistenti: servira' una procedura conservativa specifica, separata.

Dopo la sincronizzazione effettuare il collaudo personale MIG-FLOW-002.
Solo la conferma esplicita dell'utente del suo esito positivo sblocca la
migrazione successiva. Non riportare log, percorsi personali o dati host
in Git; conservare ogni evidenza reale esclusivamente in .local/.

Le modifiche ai documenti GitHub non cambiano automaticamente le istruzioni
della UI del Progetto ChatGPT: se il protocollo persistente e' stato
aggiornato, allineare anche tali istruzioni e ripetere il test di ripresa.
