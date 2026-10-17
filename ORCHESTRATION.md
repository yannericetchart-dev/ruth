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

## BRANCH-DELETE-002: rilascio utilizzatori e incarico (17 ottobre 2026)

Responsabile unico documenti comuni e integrazioni: ORCH-20261017-B.
Base letta: 7988da89891e03ce364b73ffb692db429b242bb4, PR #108 merged e CI main
39059195304 success. BRANCH-DELETE-001 resta complete: due ref gia' eliminati.
BRANCH-CLEANUP-001 e' ora active, non completa: nessuna cancellazione nuova
viene attestata prima dell'esecuzione e della verifica dei risultati.

### Conferma degli utilizzatori

L'utente ha risposto affermativamente alla domanda esplicita sul rilascio
delle vecchie chat/processi, assenza di lavori e integrazioni pendenti e
necessita dei ref: "si ti confermo che nessun processo e attivo".
Fonte: conferma utente del 17 ottobre 2026, successiva alla chiusura #108.
Il rilascio si applica individualmente a tutti i 97 branch preesistenti
congelati in BRANCH_DELETE_002.json: users_release confirmed_by_user;
writes_ceased confirmed_by_user; pending_work/integrations none_pending_confirmed_by_user.
B ha riletto 61 NODE/checkpoint: nessun altro incarico applicativo attivo,
zero PR/issue aperte alla base, nessuna integrazione pubblicata pendente.
I contratti depends_on dei componenti restano validi, ma non richiedono
la permanenza di un ref heads: main/tag/SHA preservano gli ingressi storici.
Questo rilascio deriva dalla conferma utente, non da MODEL-001 o zero PR.
Nessun nuovo perimetro funzionale e' approvato da questa conferma.

### Elenco esatto e branch della pubblicazione

BRANCH_DELETE_002.json contiene 97 coppie branch/SHA congelate: 86 tip
contenuti in main e 11 preservati da tag annotati verificati. BRANCH_AUDIT.md
ne espone la tabella; BRANCH_ARCHIVE.md conserva tag, oggetti e commit.
Il 98esimo ramo e' work/BRANCH-ARCHIVE-001-record, riutilizzato per
questa pubblicazione. B ne attesta il rilascio effettivo al merge dell'incarico:
nessun altro utilizzatore assegnato, scritture cessate dopo merge, integrazione
conclusa in main. Il suo SHA candidato esatto e' il head.sha della PR di
assegnazione merged, registrato anche letteralmente nel corpo della PR e
nel prompt esecutivo. Non usare il precedente tip osservato alla base.
Questo legame al candidato evita un riferimento del commit al proprio SHA.
L'esecutore deve verificarne la coincidenza e cancellare questo ramo per ultimo.
Dopo il rilascio B sospende nuove scritture GitHub fino al resoconto, salvo
revoca esplicita che farebbe fallire il controllo di main dell'esecutore.
Nessun nuovo branch necessario: il numero presente resta 99.

### Incarico nominativo e sequenza

ID BRANCH-DELETE-002. Esecutore: chat locale
01a14a37-aa01-70a2-b9cf-6702224b121f, "Ruth - Work Local Orchestrator".
Padre main, responsabile integrazioni B; nessun parallelismo. Sono autorizzate
solo le 97 coppie del manifest piu' l'unica coppia del ramo di pubblicazione
fissata nella PR/prompt. Protezione assoluta refs/heads/main e refs/tags/*.

Gate iniziale: PR di assegnazione merged, CI sul candidato e su main positive;
risolvere nuovamente object.sha e confrontarlo con il merge_commit_sha della PR.
Verificare inventario esatto e prerequisiti. Immediatamente prima di OGNI
rimozione rileggere main (deve restare al merge target), tip atteso, PR aperte
paginate sia head sia base, incarico, rilascio e nuove integrazioni/utilizzatori.
Per gli 11 non contenuti riconfermare il tag annotato, SHA oggetto e commit.
Su qualsiasi differenza fermarsi, senza ampliare o ricalcolare l'elenco.

Una mutazione per ramo, con condizione atomica sullo SHA atteso. E' gia'
provato dall'esecutore GitHub GraphQL updateRefs con beforeOid esatto,
afterOid 0000000000000000000000000000000000000000 e force=false.
Niente force generico, cancellazione incondizionata o selezione per prefisso.
Dopo ciascuna mutazione verificare assenza del ref. Se errore o lettura ambigua,
fermarsi e riconciliare soltanto in lettura: nessuna ripetizione automatica.
Un ramo gia' assente prima del tentativo e' absent_before_attempt; l'incarico
va riesaminato, non attribuire una cancellazione mai eseguita.

Dopo tutti i controlli: cancellare per ultimo il solo ramo documentale allo
SHA della PR, verificare inventario finale (atteso soltanto main), main
invariato e tutti gli 11 archivi con storia raggiungibile. Non scrivere nei
checkout o loro metadati, documenti, configurazioni o ambienti; nessuna sync,
copia, installazione, esecuzione Ruth o test applicativo. Log solo locali;
resoconto portabile per ogni branch/SHA/mutazione/esito, controlli PR e
rilascio, main prima/dopo, conteggi e verifica archivi. L'esecutore si ferma
e restituisce i risultati; B aggiorna gli stati effettivi dopo verifica.
I 12 NODE di componente restano present fino a rimozione verificata;
nessuno viene anticipatamente marcato archived. MIG-016 chiusa, MIG-017 non autorizzata.

## Cronologia: BRANCH-DELETE-001 chiusa alla PR #108 (17 ottobre 2026)

Responsabile comune e integrazioni: ORCH-20261017-B.
Assegnazione: PR #107 merged a 66f126c41165a012453acb1a2fd2add812d67105;
CI PR 39056771775 e CI main 39056855340 success.
Esecutore: chat locale 01a14a37-aa01-70a2-b9cf-6702224b121f,
"Ruth - Work Local Orchestrator"; resoconto inoltrato dall'utente.
Evidenza portabile: BRANCH-DELETE-001-result-001.
Stato dell'incarico: complete; risultato: passed; esecuzione delegata.

| Branch originario | SHA riconfermato prima della mutazione | Stato del ref remoto | Esito |
| --- | --- | --- | --- |
| `work/BRANCH-ARCHIVE-001-prerequisites` | `0b1ca5c3785e70eb24f503ac53ef3e3af9298e72` | archived / absent | deleted_verified |
| `work/ORCH-20261017-B-claim` | `1149ed1556e2961034a5b3d0d23ea464ac1f436e` | archived / absent | deleted_verified |

### Resoconto attribuito all'esecutore

Prima di ciascuna mutazione: rilascio e incarico confermati, integrazioni
concluse, tip atteso contenuto in main, PR aperte paginate senza
corrispondenze head/base e nessuna nuova issue.
Strumento effettivo riferito: GitHub GraphQL updateRefs, una richiesta per
ramo; beforeOid uguale allo SHA della riga, afterOid pari a
0000000000000000000000000000000000000000, force=false.
Entrambe le richieste sono state accettate. Per la seconda, la lettura
immediata era ambigua: l'esecutore si e' fermato e ha riconciliato soltanto
in lettura. REST e GraphQL hanno poi confermato entrambi i ref assenti.
Nessuna cancellazione ripetuta. L'ambiguita iniziale resta nella cronologia;
non viene omessa o registrata come seconda mutazione.
Cancellazioni effettive verificate: 2. Main prima/dopo invariato:
66f126c41165a012453acb1a2fd2add812d67105. Inventario riferito:
101 (50/50/1/0) -> 99 (50/49/0). Nessuna modifica a checkout o metadati
locali, documenti, configurazioni o ambienti; nessuna sync o esecuzione Ruth.
Capacita di cancellazione autenticata e condizionata allo SHA: provata
dall'esecutore con queste due operazioni, non una capacita mutativa di B.

### Verifica indipendente dell'orchestratore

B ha risolto nuovamente object.sha di main e verificato PR #107 merged
con le CI sopra. Inventario GitHub paginato 99/0: entrambi i nomi assenti;
matching-refs individuale per ciascun nome: lista vuota. Nessuna PR o issue
aperta. Tutti gli 11 ref heads storici, ref tags e oggetti annotati riletti:
SHA oggetto e commit dereferenziato coincidono col registro BRANCH_ARCHIVE.md.
work/BRANCH-ARCHIVE-001-record presente al tip
12e2087f5040209c01a969248121ddd052f88dec prima della presente pubblicazione.
B verifica lo stato remoto via REST; i controlli pre-mutazione e la
riconciliazione GraphQL restano attribuiti all'esecutore. B non ripete
mutazioni e non attribuisce a se' accesso o verifiche locali dell'esecutore.

### Perimetro residuo

Si chiude solo BRANCH-DELETE-001. BRANCH-ARCHIVE-001 resta complete;
BRANCH-CLEANUP-001 resta aperto e blocked_users_confirmation.
Per 97 rami preesistenti diversi da main users_release e cessazione delle
scritture restano unconfirmed: 86 contenuti e 11 storici archiviati.
MODEL-001 complete e zero PR non ne dimostrano il rilascio.
Il ramo documentale work/BRANCH-ARCHIVE-001-record e' riutilizzato da B
per questa registrazione: nessun nuovo branch creato; dopo merge resta
pending_cleanup, senza nuovo elenco di cancellazione o incarico locale.
Main e' l'unico permanente. Nessuna nuova cancellazione assegnata.
MIG-016 resta chiusa; MIG-017 e altre funzioni non autorizzate.

L'incarico originale e il rilascio esatto restano consultabili alla PR #107,
commit 66f126c41165a012453acb1a2fd2add812d67105, sezione BRANCH-DELETE-001.
L'elenco e' consumato: non autorizza ripetizioni o ulteriori rami.
Le sezioni seguenti sono la cronologia dei prerequisiti PR #106.

## Cronologia: incarico archivi e gate alla PR #106 (17 ottobre 2026)

Responsabile unico documenti comuni, mappe, PR e integrazione: ORCH-20261017-B
(claim PR #105 merged). A non ha pubblicato modifiche per questo incarico.
Base letta: 5b77b08693a808c359b294bf1ab40f4fba8f5137.

### Incarico nominativo BRANCH-ARCHIVE-001

Esecutore: chat Codex locale 01a14a37-aa01-70a2-b9cf-6702224b121f,
titolo attuale "Ruth - Work Local - Orchestrator".
Ruolo limitato: esecutore locale; il titolo non attribuisce titolarita
di orchestratore, documenti comuni o mappe.
Incarico esplicitamente richiesto dall'utente nel perimetro gia' approvato.
Responsabile del padre e delle integrazioni: ORCH-20261017-B; padre main.
Nessun branch applicativo assegnato, nessun parallelismo autorizzato.
Primo gate: merge della PR di questi prerequisiti con CI positiva.
Dopo il gate, unico lavoro consentito all'esecutore: creare e verificare
gli 11 tag duraturi esattamente elencati in BRANCH_ARCHIVE.md.
Nessuna mutazione e' attestata come gia' riuscita.

Prima di ciascun tag rileggere main e l'esatto tip remoto del ramo;
confrontare con il registro e fermarsi su differenze, contese o mutamenti
degli incarichi. Riconfermare PR head/base e stato degli utilizzatori noto.
Preferire tag annotati; nessun tag di prova, sovrascrittura, spostamento
o eliminazione di tag. Se il nome esiste, verificarlo senza riscriverlo;
se non coincide con il commit atteso, fermarsi e riferire il conflitto.
Verificare il ref remoto e dereferenziare l'eventuale oggetto annotato
fino al commit atteso. Conservare immutabili gli archivi verificati.
Restituire per ogni riga: branch, tip riconfermato, tag, tag-object-SHA
(o non_applicable se leggero), commit dereferenziato, esito ed eventuale blocco.
Su fallimento o esito ambiguo fermarsi e riportare anche l'eventuale oggetto
creato senza ref; non attestare archivio completo da un solo tag object.
Dopo gli archivi FERMARSI prima delle cancellazioni e delle modifiche
documentali: il resoconto torna a B. Nessuna scrittura locale ai checkout,
sync, installazione, nuova copia, esecuzione Ruth o test applicativo.

### Evidenze e attribuzione

Resoconto dell'esecutore, inoltrato su istruzione esplicita dell'utente:
main e titolarita B verificati sulla base sopra; checkout designato con
remoto canonico, main, HEAD coincidente e working tree pulito, invariato.
Lettura ordinaria disponibile; con accesso esteso controlli dei permessi
indicano scrittura disponibile. La prova temporanea creazione/lettura/
rimozione appartiene all'incarico precedente, non e' stata ripetuta qui.
Autenticazione gh e letture API riuscite; push=true dichiarato dall'API.
Queste sono verifiche dell'esecutore, non accessi locali eseguiti da B.
Il pending_no_access della sessione cloud B non e' una diagnosi del checkout.
Dopo il merge documentale, HEAD locale riferita resta quella della base;
nessuna sync viene richiesta da questo incarico di soli ref remoti.

Inventario dell'esecutore paginato 50/49/0: 99 branch, main protetto,
87 contained_candidate, 11 archive_required; zero PR/issue aperte e zero tag.
Gli 11 tip e nomi tag coincidono con BRANCH_ARCHIVE.md alla base.
61 NODE/checkpoint esaminati; nessun incarico locale pubblicato trovato.
Nessun controllo tecnico fallito. create-tag e delete-ref NON TENTATE:
push=true non prova la loro capacita effettiva.
Tag creati 0, branch cancellati 0, documenti/PR modificati dall'esecutore 0.

Verifiche dirette B in questa pubblicazione: ref main e PR #105 merged,
61 NODE/checkpoint letti alla base, 99 branch (seconda pagina vuota),
zero PR/issue aperte e zero tag, tutti gli 11 ref heads letti individualmente
e coincidenti con il registro. Il nuovo branch documentale e' un ulteriore
temporaneo, utilizzato da B fino al merge e poi pending_cleanup.
Nessuna mutazione tag o cancellazione eseguita da B.

### Utilizzatori, fonti e integrazioni pendenti

MODEL-001 viene chiusa per riallineare lo stato del catalogo CAT-001 gia'
completo e accettato; Models implementation resta partial. Evidenze storiche
CAT-001 e CAT-MANUAL-001 preservate. La chiusura NON certifica rami liberi.
PR aperte head/base: zero al controllo. Le integrazioni applicative
approvate risultano concluse; MIG-016 resta chiusa con prova delegated,
MIG-017 resta soltanto proposta. Nessuna integrazione storica non approvata
viene attivata o importata per facilitare la pulizia.

Per gli undici rami: nessun incarico operativo pubblicato concorrente
individuato nei 61 NODE/checkpoint e nei registri comuni consultati.
Non e' disponibile una conferma nominativa che tutti i precedenti
utilizzatori abbiano cessato le scritture; l'assenza di PR o di un incarico
pubblicato non la sostituisce. Stato per ciascuno:
users_release: unconfirmed; deletion_gate: blocked_users_confirmation.
Le fonti storiche Python e conversazione sono tuttora utilizzate come
riferimenti di migrazione: richiedono anche aggiornamento operativo da B
dopo prova dei tag, mantenendo nomi originari e SHA nella cronologia.
Le note pending storiche restano cronologia; nessuna integrazione nuova
viene dedotta da esse.

Secondo gate: B riceve e controlla gli 11 esiti, verifica gli archivi remoti
e aggiorna i riferimenti operativi e il registro via PR/CI/merge.
Prima di qualunque rilascio per cancellare, risolvere users_release per
ciascun ramo, riconfermare incarichi e integrazioni pendenti e PR head/base.
B pubblichera' solo allora un elenco esatto branch/SHA eliminabili con
riferimento al merge documentale. Nessun elenco eliminabile rilasciato ora.
Terzo gate: cancellazione soltanto con condizione sullo SHA atteso e nuova
verifica immediata di main, tip, PR, utilizzatori e integrazioni.
Se la mutazione disponibile non supporta quella condizione, fermarsi.
I checkout storici restano intatti; main unico branch permanente.
