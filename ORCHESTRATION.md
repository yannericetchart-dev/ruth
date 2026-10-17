# Ruth: orchestrazione e continuita' fra chat

Protocollo CONT-001 revisione 2; decisione ORCH-001 revisione 2.
HANDOFF-001 e BRANCH-LIFECYCLE-001 approvate il 17 ottobre 2026.
Approvazione esplicita dell'utente: 17 ottobre 2026.
MIG-FLOW-002 revisione 3 disciplina il collaudo delegato.
Questo documento non approva nuove funzioni o migrazioni applicative.

## Ruoli e attivazione

| Ruolo | Responsabilita |
| --- | --- |
| Utente | Approva ogni nuovo perimetro funzionale e le attivita autorizzate al parallelismo; apre e fa proseguire le chat branch. |
| Orchestratore Work Cloud, fuori dal progetto | Ricostruisce lo stato GitHub, prepara incarichi e ordine dei prerequisiti, coordina le integrazioni; e' responsabile unico dei documenti comuni e dell'integrazione in main. Esegue sync e collaudo locali post-merge quando accesso, permessi e delega sono verificati. |
| Chat branch ChatGPT ordinaria, dentro il progetto Ruth | Responsabile unico del branch assegnato; implementa il contratto approvato, pubblica il proprio checkpoint e propone aggiornamenti comuni all'orchestratore. |
| Responsabile del padre | Unico soggetto assegnato alle integrazioni in quel padre; verifica contratti, revisioni e CI e integra serialmente. Puo' coincidere con l'orchestratore. |

Le chat ordinarie sono aperte e fatte proseguire dall'utente. Non presumere
creazione, messaggistica o prosecuzione autonoma fra chat, condivisione di
cronologia o strumenti locali nelle chat branch. L'orchestratore prepara
consegne e messaggi di ripresa; GitHub e' il canale persistente di stato.
Ogni chat verifica i propri strumenti GitHub prima di assumere l'incarico.
Il nome Work Cloud descrive il ruolo organizzativo, non dimostra la sede
dell'esecutore o l'accesso alla macchina. Gli accessi si verificano per risorsa.

## Verde iniziale e incarico

Prima di implementare, il perimetro deve descrivere scopo, API, ingressi,
risultati, controlli, effetti sui dati, limiti, percorso dall'ingresso Ruth,
esiti attesi e test prescritti. Dichiarare subito le eventuali verifiche
che richiedono una valutazione umana, con criteri e momento di esecuzione.
Il verde identifica le attivita che possono procedere in parallelo, i loro
responsabili, i confini delle modifiche e le integrazioni seriali previste.
Se un'attivita non e' esplicitamente inclusa nel parallelismo, non avviarla
in parallelo. Non attivare automaticamente altri componenti.

Ogni incarico pubblica nel checkpoint del componente un contratto con:

- ID stabile e riferimento al perimetro e alla sua approvazione;
- responsabile unico, branch di lavoro, padre e responsabile del padre;
- SHA base e revisione delle istruzioni/contratti consultate;
- API e responsabilita, file/componenti assegnati e limiti degli effetti;
- prerequisiti con ID, produttore, stato richiesto e revisione/evidenza;
- criteri di completamento, CI/test richiesti e valutazioni umane previste;
- SHA candidato, PR, risultati portabili, blocchi e prossimo passo.

Il passaggio di responsabilita va registrato esplicitamente prima di nuove
scritture. Le chat branch aggiornano i propri checkpoint e metadati assegnati;
le proposte per decisioni, indice, checkpoint radice e altri documenti comuni
passano al loro responsabile unico. La rigenerazione globale delle mappe e'
serializzata dallo stesso responsabile, evitando modifiche concorrenti alle
viste generate. Nessun dato di host entra nei contratti (DATA-001).

## Prerequisiti e assenza di attese circolari

Distinguere contratto disponibile, implementazione verificata e integrazione
nel padre. Ogni prerequisito ha un produttore e un criterio verificabile:
un generico stato active o il nome del branch non bastano per sbloccarlo.
Un incarico e' pronto solo con tutti i prerequisiti soddisfatti; altrimenti
registrare ID bloccante, responsabile e azione di sblocco.

Prima di avviare il gruppo, l'orchestratore controlla l'intero grafo delle
attivita e ordina i prerequisiti senza cicli. Un figlio non puo' attendere
l'integrazione finale del padre che deve riceverlo: se gli serve un contratto
del padre, definirlo con un'attivita preliminare separata e una revisione
condivisa prima dell'implementazione dipendente.
Se emerge un ciclo, fermare solo le attivita coinvolte, estrarre il
prerequisito comune o stabilire una sequenza esplicita e aggiornare i
checkpoint; un nuovo perimetro richiede il verde dell'utente.

NODE.yaml mantiene gli ID stabili, depends_on e blocked_by secondo lo
schema esistente. Il validatore attuale verifica i cicli depends_on fra
nodi e l'esistenza degli ID blocked_by; non certifica da solo l'assenza di
cicli fra tutti gli incarichi. Il controllo di questi ultimi e' documentale
e responsabilita dell'orchestratore. Questa revisione non modifica il
validatore o altri programmi.

## Pubblicazione e integrazioni seriali

Per ogni modifica versionata usare GitHub: branch di lavoro, commit, PR,
revisione e CI. Il checkout personale non e' una sorgente di sviluppo.
Un solo responsabile integra in ciascun padre, una PR alla volta, lungo
la catena dichiarata dal componente fino a main.
Prima del merge verificare SHA del candidato e del padre, contratto,
prerequisiti, diff, DATA-001, checkpoint e CI pertinenti.
Se cambia il contenuto candidato o il padre influenza la verifica,
riesaminare e verificare la combinazione risultante prima di integrare.
Non considerare CI di una revisione diversa come prova del candidato.

Il verde funzionale comprende le promozioni ordinarie fino a main.
L'avanzamento non richiede ulteriori verdi per ogni padre, ma non permette
all'orchestratore di avviare autonomamente le chat branch ordinarie.
Tenere separati codice candidato, verifica CI, integrazione, sync locale,
collaudo delegato e valutazione umana quando prevista.

## Sync locale post-merge

Dopo merge in main, fissare nuovamente il commit target e verificare in
lettura la copia pulita esplicitamente designata: remoto canonico, main,
HEAD, file tracciati/non tracciati e permessi effettivi della sessione.
Una shell o un percorso Linux non dimostrano accesso locale o scrittura.
Verificare sia i permessi del sistema operativo sia le radici e restrizioni
della sessione; un controllo read-only non e' una prova di scrittura.

Aggiornare soltanto con permessi sufficienti, copia pulita sul branch main,
remoto corretto e fast-forward possibile verso lo SHA target verificato.
Confrontare il ref remoto acquisito con quel target e poi l'HEAD risultante;
se il target si muove, fermare l'aggiornamento e rivalutare la revisione.
Non usare reset, clean, stash, rebase, force, cambio branch o scarti per
nascondere divergenze. Preservare integralmente copie storiche e modifiche.

Senza permessi, registrare pending_permissions; senza collegamento,
pending_no_access; con copia sporca, su altro branch o divergente,
blocked_divergence. Il blocco locale non annulla l'integrazione remota,
ma impedisce di dichiarare il collaudo locale completato.
Non trasformare automaticamente un blocco in lavoro personale dell'utente.
Una nuova copia o un nuovo ambiente richiedono autorizzazione specifica:
nessuna creazione o importazione dalla copia storica e' implicita.

## Collaudo delegato e chiusura: MIG-FLOW-002 revisione 3

Dopo l'integrazione in main, l'orchestratore puo' eseguire il collaudo locale
delegato sulla copia autorizzata e sulla revisione verificata. L'esecuzione
comprende il punto d'ingresso ufficiale senza argomenti, l'intero flusso
integrato nell'ordine atteso, gli esiti prescritti e tutti i test mirati
indicati dal perimetro. Un import, una API isolata o la sola CI non bastano.

Il collaudo delegato completo e positivo soddisfa il gate locale e consente
la chiusura della porzione: non richiedere all'utente di ripetere operazioni
o dare un secondo consenso. Registrare sempre mode: delegated; non
attribuire mai l'esecuzione personale all'utente.
Un eventuale collaudo eseguito davvero dall'utente conserva mode: personal.
Le accettazioni personali storiche non vengono rinominate o reinterpretate.

Se erano previste valutazioni umane, la chiusura attende anche il loro esito.
Esse devono essere dichiarate nel perimetro iniziale; nuove esigenze emerse
vanno esplicitate e risolte con l'utente, mai considerate automaticamente
soddisfatte o introdotte di nascosto come duplicazione del collaudo.

Prove incomplete, fallite, ambigue o incongruenti lasciano la porzione aperta.
Correggere entro lo scope, ripubblicare su GitHub e ripetere le verifiche
necessarie sulla revisione risultante, senza avanzare alla migrazione seguente.
not_configured e' uno skip valido solo quando previsto: non certifica GGUF
reali, pesi, CUDA, backend o inferenza mai verificati.

Dopo esito completo, chiudere autonomamente checkpoint, NODE e mappe tramite
PR/CI/merge e proporre il perimetro successivo. Prima di implementarlo
attendere il nuovo OK esplicito dell'utente (MIG-FLOW-001).
La chiusura documentale successiva non invalida da sola il collaudo dello
SHA provato: verificare che il diff sia solo documentale, conservarne il
riferimento e indicare pending_docs_fast_forward fino alla sync dei documenti.
Se cambia codice o contenuto che influenza il collaudo, rivalutare le prove.

## Registrazione portabile delle evidenze

Registrare separatamente commit applicativo/integrato, commit collaudato,
stato di sync e identificativo del collaudo. Per il collaudo indicare mode
(delegated/personal), status (pending/passed/failed/incomplete/blocked),
fasi/esiti attesi soddisfatti, conteggio test e valutazione umana
(required oppure not_required; pending/passed quando richiesta).
La chiusura resta pending finche' manca una verifica prescritta.

Questi campi descrivono checkpoint documentali, non un nuovo schema runtime
o un'estensione del validatore NODE. In Git soltanto SHA, stati, conteggi e
riferimenti portabili a PR/CI. Log, percorsi personali, inventari, versioni
specifiche di pacchetti, credenziali, manifest e dati reali restano locali.

## Sostituzione dell'orchestratore (HANDOFF-001)

Prima del cambio chat, il titolare pubblica su GitHub una consegna completa:
ID del passaggio, responsabile uscente/entrante, perimetri e approvazioni
ancora validi, SHA di main e delle prove, incarichi con titolari, branch e
PR, prerequisiti, verifiche, blocchi e prossima azione.
La consegna e' in ORCHESTRATOR_HANDOFF.md; le coordinate personali delle
risorse locali restano fuori da Git e sono fornite privatamente dall'utente.
Le approvazioni valide restano valide; il cambio chat non approva nuovi scope.

Fasi: preparing -> released -> claimed. Durante preparing l'uscente esegue
soltanto le scritture necessarie a completare questa consegna; non avvia
nuovi incarichi. Il rilascio diventa effettivo al merge della PR di consegna
in main. Da quel momento l'uscente cessa nuove scritture ai documenti comuni,
anche se continua a leggere o a rispondere. Il candidato successore non
scrive sui documenti comuni prima della presa in carico, salvo la PR
dedicata alla sola registrazione della presa in carico descritta sotto.

Il successore Work Cloud opera fuori dal progetto: risolve nuovamente
heads/main e legge a quello SHA indice, decisioni, checkpoint, consegna,
audit/archivi, incarichi e mappe. Ricostruisce lo stato da GitHub, non dalla
sola cronologia o dal messaggio di avvio. Verifica nuovamente strumenti
autenticati, accessi locali per risorsa, permessi della sessione e ambiente.
Una verifica storica non dimostra accessi o scrittura nella nuova chat.
In assenza di risorsa/permesso dichiara il blocco, senza cercare credenziali,
creare ambienti o modificare checkout implicitamente.

La presa in carico e' esplicita: pubblicare tramite PR il proprio ID
organizzativo, l'ID della consegna, il precedente titolare e lo SHA della
consegna letta, confermando le verifiche eseguite e i blocchi rimasti.
La PR di claim modifica solo lo stato della consegna e il checkpoint
necessario ad attribuire la responsabilita; non avvia altro lavoro.
Prima del merge rileggere consegna, owner e main: se e' gia' avvenuto
un claim o c'e' contesa tra successori, non integrare un claim obsoleto
e non procedere con altre scritture. Far risolvere la contesa all'utente.
Una sola presa in carico confermata abilita il nuovo titolare alle
scritture comuni. Non presumere una presa in carico dal silenzio.

Il testo completo di avvio e' in SUCCESSOR_MESSAGE.md. L'utente puo'
aprire personalmente la chat. Non presumere la creazione automatica se
gli strumenti dell'app non rispondono; la sola esposizione di un tool
non equivale a una chat creata o a un passaggio completato.

## Branch temporanei (BRANCH-LIFECYCLE-001)

main e' l'unico branch permanente. Creare solo i rami necessari al lavoro
approvato, inclusi eventuali padri temporanei richiesti dalle integrazioni.
Il nome /main di un componente indica il suo ruolo nell'integrazione,
non una promessa di permanenza. Una mappa planned non autorizza a creare
il branch. Ogni ramo temporaneo ha responsabile, utilizzatori, perimetro,
padre, integrazioni pendenti e criterio di rimozione nel checkpoint.
Un branch senza queste verifiche non e' automaticamente eliminabile.

Il responsabile unico della pulizia e' l'orchestratore titolare. Prima
di OGNI cancellazione rileggere main, SHA corrente del ramo, PR aperte sia
con quel ramo come head sia come base, incarichi/utenti e integrazioni
pendenti. Confrontare con lo SHA esaminato; se cambia, riesaminare e non
cancellare sulla base di un inventario precedente. Confermare che il
titolare del branch abbia cessato le scritture e non vi siano utilizzatori
attivi. Una PR merged o ahead_by=0, da soli, non provano che il ramo sia libero.

Se tutta la storia e' contenuta in main e non vi sono utilizzatori o
integrazioni pendenti, il ramo puo' essere eliminato nel perimetro autorizzato.
Se vi sono commit non contenuti, PRIMA preservare l'esatto tip con un tag
remoto duraturo d'archivio; registrare ramo originario, commit e tag in
BRANCH_ARCHIVE.md. Preferire un tag annotato; verificare il ref remoto e,
se annotato, dereferenziarlo fino al commit atteso. Il tag resta immutabile:
non spostarlo o eliminarlo durante la pulizia. Un nome pianificato o uno
SHA scritto in un documento non sostituiscono un ref d'archivio esistente.

Dopo verifica del tag aggiornare i riferimenti correnti di migrazione
al tag e allo SHA immutabile, conservando i nomi storici nella cronologia.
Riconfermare subito prima della cancellazione SHA, PR e incarichi.
Usare strumenti autenticati per la mutazione richiesta e, quando disponibile,
una condizione sullo SHA atteso; poi rileggere il remoto per confermare
l'assenza del ramo. Mai importare codice storico in main per rendere
eliminabile un ramo; la storia non integrata resta nell'archivio.

Dopo una rimozione verificata aggiornare checkpoint, NODE e mappe: un ramo
eliminato non puo' restare present. Usare archived per il ramo dismesso,
mantenendo nome/padre logici e prova della conservazione; planned indica
un ramo futuro non ancora creato. Flag temporary_branch identifica i rami
di componente oggi presenti; solo main ha permanent_branch.
I padri logici nei metadati possono non esistere come ref: ricrearli solo
quando necessari a un lavoro approvato, registrando lo stato reale.

Questa decisione sostituisce esplicitamente la precedente permanenza dei
branch storici: si conserva la storia tramite archivi verificati, non
necessariamente il ref heads originario. Non riscrivere la cronologia.
La pulizia riguarda ref remoti; non autorizza modifiche ai checkout storici.

Se mancano creazione tag o delete-ref autenticati, dichiarare il blocco
concreto e lasciare i branch intatti. Non simulare tag con branch, non
usare create_branch/update_ref come strumenti per tag o cancellazioni
e non trattare GET come una mutazione. Non dichiarare completa la pulizia.
BRANCH_AUDIT.md contiene uno snapshot di audit, da riconfermare live.
