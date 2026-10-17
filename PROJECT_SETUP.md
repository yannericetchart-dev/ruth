# Ruth: configurazione e collaudo della continuita'

Protocollo CONT-001 revisione 2; ORCH-001; MIG-FLOW-002 revisione 3.
Fonte comune: GitHub, branch main. Procedura generica senza dati di macchina.

## Ripresa e ruoli

A ogni ripresa risolvere il ref heads/main tramite il connettore GitHub,
usare object.sha per START_HERE.md e tutti i documenti applicabili.
Leggere AGENTS.md, DECISIONS.md, CHECKPOINT.md, ORCHESTRATION.md e mappe
del componente e antenati. Non usare un blob SHA come commit.
In mancanza di accesso o fonti dichiarare la ripresa incompleta.

L'orchestratore Work Cloud opera fuori dal progetto. Le chat branch sono
ChatGPT ordinarie dentro il progetto Ruth, aperte e fatte proseguire
dall'utente. Gli strumenti effettivi si verificano in ogni chat.
GitHub conserva contratti, prerequisiti e checkpoint; la cronologia delle
altre chat non e' un canale di coordinamento presunto.
Assegnazioni, parallelismo esplicitamente approvato, responsabili unici e
integrazioni seriali seguono ORCHESTRATION.md.

## Sviluppo GitHub-first (REMOTE-001)

Tutte le modifiche versionate avvengono su GitHub con branch, commit, PR,
revisioni e CI fino a main. Il checkout principale resta intatto durante
lo sviluppo; non importare o eliminare modifiche locali preesistenti.
Gli strumenti GitHub mancanti sono un blocco, non una ragione per sviluppare
silenziosamente nel clone personale.

Le verifiche pre-merge possono usare solo snapshot puliti del candidato
pubblicato, in spazi isolati autorizzati, ignorati o esterni al checkout.
Le copie, gli ambienti e le installazioni non sono autorizzati implicitamente.
Correzioni versionate vanno ripubblicate su GitHub e nuovamente verificate.
Cache, build, fixture temporanee e risultati restano fuori dai file tracciati.

## Sincronizzazione finale delegata

La sincronizzazione non e' un prerequisito dello sviluppo remoto.
Dopo il merge in main l'orchestratore fissa lo SHA target dal ref GitHub e
controlla in lettura il checkout pulito designato, senza cercare o alterare
copie storiche. Comandi diagnostici possibili, nella copia autorizzata:

```sh
GIT_OPTIONAL_LOCKS=0 git remote -v
GIT_OPTIONAL_LOCKS=0 git branch --show-current
GIT_OPTIONAL_LOCKS=0 git rev-parse HEAD
GIT_OPTIONAL_LOCKS=0 git -c core.fsmonitor=false status --porcelain=v1 --untracked-files=all --ignore-submodules=none
```

Verificare separatamente permessi del sistema operativo e limiti di scrittura
della sessione su checkout, metadati Git e destinazioni di output.
La leggibilita' e la presenza di .venv non dimostrano scrittura o prontezza
dell'ambiente. Non fare prove di scrittura fuori dall'autorizzazione.

Solo con permessi sufficienti, remoto canonico, main pulito e fast-forward
possibile, acquisire il ref remoto dopo il merge e confrontarlo con lo SHA
target. Se coincide, procedere con merge --ff-only verso lo SHA fissato,
poi verificare HEAD e pulizia. Nessun aggiornamento a una revisione diversa
senza nuova valutazione del target.

Fermarsi con pending_permissions, pending_no_access o blocked_divergence
quando applicabile. Non usare reset, clean, stash, rebase, force, cambio
branch o scarto file automatico. Il checkout storico modificato va
preservato integralmente. Nuove copie/ambienti richiedono autorizzazione
separata; non copiare automaticamente modifiche o dati dalla copia storica.

## Collaudo locale delegato (MIG-FLOW-002 revisione 3)

Dopo la sync verificata, usare l'interprete della copia pulita designata,
senza installazioni implicite, per il default main.py e i test prescritti.
Verificare revisione, intero flusso, ordine e riepilogo, esiti attesi e
tutti i test previsti dal perimetro. Conservare output ed evidenze reali
solo localmente in spazi autorizzati; in Git soltanto SHA, stati e conteggi.
Evitare output o cache nei file tracciati e ricontrollarne la pulizia.

Il collaudo delegato completo e positivo puo' chiudere la porzione senza
ripetizione personale o conferma ridondante. Registrare mode: delegated,
mai esecuzione personale dell'utente. L'esecuzione personale effettiva resta
registrabile come personal; le prove storiche mantengono la loro attribuzione.
Valutazioni umane sono gate solo se dichiarate nel perimetro iniziale.
Prove incomplete, fallite o incongruenti mantengono aperta la porzione.

Dopo la chiusura verificata aggiornare documentazione/mappe via PR/CI/merge,
proporre lo scope seguente e attendere il nuovo OK prima di implementarlo.
Se la chiusura modifica solo documenti, mantenere il riferimento al commit
collaudato e registrare pending_docs_fast_forward per la sync residua;
non attribuire un'esecuzione alla nuova revisione senza averla svolta.

## Testo da inserire nelle istruzioni del progetto ChatGPT

Conservare le istruzioni precedenti compatibili e sostituire le indicazioni
operative incompatibili con questo blocco. La modifica della UI e' distinta
dalla pubblicazione GitHub; non e' stata eseguita automaticamente.

```text
Progetto: yannericetchart-dev/ruth. CONT-001 rev.2, ORCH-001, MIG-FLOW-002 rev.3.
A ogni nuova chat o ripresa risolvi prima main tramite la connessione GitHub:
https://api.github.com/repos/yannericetchart-dev/ruth/git/ref/heads/main
Usa object.sha come ref per START_HERE.md, AGENTS.md, DECISIONS.md,
CHECKPOINT.md, ORCHESTRATION.md e le mappe del componente e dei suoi antenati.
Lo SHA di un blob non e' il commit. Se manca una fonte, dichiara il blocco.
Questa e' una chat branch ChatGPT ordinaria nel progetto Ruth, aperta e fatta
proseguire dall'utente. L'orchestratore Work Cloud opera fuori dal progetto.
GitHub contiene istruzioni, contratti, checkpoint e dipendenze condivisi.
Prima del lavoro verifica incarico, responsabile unico, branch/padre, SHA base,
scope approvato e prerequisiti. Il verde iniziale deve elencare esplicitamente
le attivita autorizzate anche al parallelismo; nessuna concorrenza implicita.
Rispetta il responsabile unico delle integrazioni in ciascun padre e quello
dei documenti comuni. Pubblica il checkpoint del tuo incarico e proponi al
responsabile gli aggiornamenti comuni, senza scritture concorrenti.
Modifiche versionate solo su GitHub; promozioni seriali e CI fino a main.
Preserva i checkout locali durante lo sviluppo e ogni lavoro preesistente.
Aggiornamento conservativo e collaudo locali post-merge sono delegati
all'orchestratore, solo con accessi e permessi effettivamente verificati.
Il collaudo delegato completo e positivo puo' chiudere la porzione: SHA,
flusso ufficiale completo, esiti attesi e test prescritti devono coincidere.
Registralo come delegated, mai come esecuzione personale dell'utente.
Prove incomplete o fallite mantengono aperta la porzione. Valutazioni umane
sono richieste solo se dichiarate nel perimetro iniziale.
Dopo chiusura verificata proponi lo scope successivo e attendi il nuovo OK
prima di implementarlo. Non dedurre autorizzazioni dal silenzio.
DATA-001: in Git solo materiale portabile, SHA, stati e conteggi; nessun
percorso personale, log, inventario host, dato reale o credenziale.
Kaggle resta inattivo. I documenti GitHub non aggiornano da soli la UI.
```

## Collaudo indipendente della continuita'

Dopo l'allineamento UI, l'utente apre una nuova chat ordinaria nel progetto.
Messaggio di prova: Riprendi il progetto Ruth.
Senza introdurre codice, verificare risoluzione dello SHA, ricostruzione
dello stato, incarico e prerequisiti, ruoli, parallelismo autorizzato,
DATA-001 e distinzione fra CI, integrazione, sync e collaudo delegato.
Ripetere la ripresa nell'orchestratore fuori dal progetto; non presume
le istruzioni UI del progetto e deve leggere GitHub.
Registrare solo esiti generali portabili. Non dichiarare verificata ogni
chat futura sulla base di un collaudo storico.
