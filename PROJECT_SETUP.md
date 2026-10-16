# Ruth: configurazione e collaudo della continuita'

Protocollo: CONT-001, revisione 1. Fonte comune: GitHub, branch `main`.
Questo documento e' una procedura generica, senza percorsi o account locali.

## Codex

Aprire il clone effettivo come cartella principale del progetto. Verificare
che contenga `AGENTS.md` e che il remote corrisponda al repository canonico.
Il checkout locale deve essere confrontato con la revisione GitHub letta;
non cambiare branch o sovrascrivere modifiche locali per nascondere divergenze.
Le istruzioni degli agenti richiamano START_HERE.md; non affidarsi alla memoria.

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
Confronta il checkout locale con la revisione pubblicata quando disponibile.
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

### Sincronizzazione del checkout dopo un merge in main (SYNC-001)

La pubblicazione su GitHub non aggiorna automaticamente un clone sul computer.
Se il client ha accesso esplicito alla cartella locale, prima di operare
confrontare GitHub e il clone, senza assumere che il branch attivo sia main.
Eseguire dalla radice del clone, senza copiare in Git i risultati locali:

```sh
git remote -v
git status --short --branch
git branch --show-current
git rev-parse HEAD
git fetch origin
git rev-parse origin/main
```

Confermare che `origin` punti a yannericetchart-dev/ruth e che `origin/main`
coincida con il commit main pubblicato, letto mediante il connettore GitHub.
Solo se il checkout e' pulito, il branch attivo e' `main` e non ci sono
modifiche o lavori concorrenti da preservare, aggiornare senza merge impliciti:

```sh
git merge --ff-only origin/main
git rev-parse HEAD
git status --short --branch
```

Il valore HEAD finale deve coincidere con il commit GitHub atteso. In caso
di branch differente, modifiche non committate, merge non fast-forward,
remote non atteso o incertezza, interrompere la sincronizzazione automatica
e registrare il motivo. Non scartare o nascondere modifiche dell'utente,
non usare reset --hard, clean, rebase, stash o force-push come scorciatoie.

Se il client non puo' aprire la cartella reale, riportare in CHECKPOINT.md
`local_sync: pending_no_access` e commit GitHub, poi fornire la procedura
per eseguirla tramite un ambiente autorizzato (Work desktop locale,
Codex locale o l'utente). Non riportare percorsi assoluti, output dettagliati
o dati host in Git. Il collaudo manuale previsto da MIG-FLOW-002 resta
obbligatorio anche se la sincronizzazione Git e' riuscita.
