# Messaggio completo per avviare l'orchestratore successore

Nota di ripresa corrente: dopo lettura di main/START_HERE e della consegna,
controllare l'ultimo checkpoint valido BRANCH-CLEANUP-001-FINAL nella PR di registrazione
con head work/BRANCH-CLEANUP-001-final-record, insieme a main live e agli
11 archivi. Gli stati pending del NODE sono snapshot della pubblicazione;
solo un commento finale complete verificato sostituisce il gate residuo.


Da inviare a una nuova chat Work Cloud fuori dal progetto Ruth.
Le coordinate personali delle risorse locali vanno aggiunte privatamente;
non inserirle nei documenti GitHub. Nessuna nuova migrazione e' autorizzata.

```text
Assumi il ruolo di orchestratore successore del repository
yannericetchart-dev/ruth, in Work Cloud fuori dal progetto.

La consegna autorizzata e' RUTH-HANDOFF-20261017-01. Prima di qualsiasi scrittura comune:
1. Usa la connessione GitHub per risolvere
   https://api.github.com/repos/yannericetchart-dev/ruth/git/ref/heads/main
   e usa object.sha come ref. Non usare uno SHA blob come commit.
2. Leggi a quello SHA START_HERE.md, AGENTS.md, DECISIONS.md, CHECKPOINT.md,
   ORCHESTRATION.md, ORCHESTRATOR_HANDOFF.md, BRANCH_AUDIT.md,
   BRANCH_ARCHIVE.md, PROJECT_SETUP.md, ARCHITECTURE.md, MIGRATION.md
   e metadati/mappe applicabili. Ricostruisci branch, PR, incarichi,
   prerequisiti, verifiche e blocchi direttamente da GitHub.
3. Verifica che la PR di consegna sia merged, l'uscente sia released e
   non vi sia gia' un successore titolare o un claim concorrente.
   L'uscente ORCH-20261017-A ha cessato nuove scritture comuni al rilascio.
4. Verifica nuovamente gli strumenti autenticati disponibili. Distingui
   esecuzione cloud, risorse locali e connessione GitHub. Per ogni risorsa
   locale verifica accesso, permessi effettivi della sessione e ambiente.
   Non dedurre accesso dal nome Linux, da una shell o dalle prove della
   chat precedente. Chiedi solo le coordinate locali mancanti; non
   cercare indiscriminatamente file o credenziali.
5. Pubblica una PR di sola presa in carico con il tuo ID organizzativo,
   handoff_id, precedente titolare, SHA della consegna letta, verifiche
   e blocchi. Riconferma prima del merge che il claim non sia obsoleto.
   In caso di contesa fermati. Solo dopo il claim merged assumi
   esplicitamente l'unica responsabilita delle scritture comuni.

Mantieni CONT-001 rev.2, ORCH-001 rev.2, HANDOFF-001, BRANCH-LIFECYCLE-001,
MIG-FLOW-001 e MIG-FLOW-002 rev.3. Le approvazioni valide restano valide;
il cambio chat non ne estende il perimetro. Sviluppo GitHub-first,
DATA-001 e checkout storici intatti. Chat branch ChatGPT ordinarie nel
progetto Ruth, aperte e fatte proseguire dall'utente. Parallelismo solo
per attivita identificate nel verde; integrazioni seriali e titolari unici.

MIG-016 e' chiusa con collaudo delegated sul commit
dbdef7ab000d99d2e3dde68b0857840f0047d5fe: Ruth 13 fasi, quattro fixture
sintetiche passed e 11 test OK. Chiusura PR #103, main precedente
c1d3572627b0f544ae6ef54d05daca7ee2b62f62. Non attribuire una prova personale all'utente.

MIG-017, fixture multidimensionali, e' soltanto proposta: NON implementarla,
non avviare nuove funzioni e non assegnare nuovo lavoro applicativo senza OK.

Riprendi la pulizia remota gia' autorizzata solo dopo il claim e se disponi
delle mutazioni autenticate necessarie. main e' l'unico branch permanente.
L'audit base aveva 97 branch: 85 contenuti e 11 non contenuti, zero PR aperte
e zero tag. Riconferma tutto live; esiste anche il ramo della consegna.
Per ogni cancellazione verifica SHA, main, PR head/base, incarichi,
utilizzatori e integrazioni pendenti. MODEL-001 e' ancora active nei metadati.
Per i rami non contenuti crea e verifica PRIMA tag remoti d'archivio duraturi,
aggiorna i riferimenti di migrazione e solo dopo elimina i branch.
Nessun codice storico da importare in main. Se mancano strumenti o
autenticazione dichiara il blocco: finora zero cancellazioni e zero tag.

Le risorse locali esistenti hanno stati distinti. La copia dedicata era
pulita al main precedente; le altre copie non sono dichiarate sincronizzate.
Non creare nuove copie, ambienti o installazioni implicitamente.
Dopo la presa in carico riporta revisione, titolarita assunta, strumenti
verificati, stato reale delle risorse, blocchi e prossima azione autorizzata.
```
