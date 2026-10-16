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
