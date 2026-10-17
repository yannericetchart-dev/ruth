# Ruth: consegna dell'orchestratore

handoff_id: RUTH-HANDOFF-20261017-01
state: released_on_merge
outgoing_owner: ORCH-20261017-A
incoming_owner: unassigned
source_main_sha: c1d3572627b0f544ae6ef54d05daca7ee2b62f62
publication_branch: work/ORCH-SUCCESSION-BRANCH-LIFECYCLE
publication_pr: #104 (https://github.com/yannericetchart-dev/ruth/pull/104)
prepared_snapshot_sha: 16cc38f374707639ec90b3ed066bcca066889f3e
publication_sha: leggere il merge_commit_sha della PR indicata, non il blob SHA
common_documents_writer_after_release: none_until_verified_claim
writes_frozen_after_publication_merge: true

Questa consegna e' preparata dall'orchestratore uscente su autorizzazione
dell'utente. Il rilascio diventa effettivo soltanto quando la PR e' merged
in main. Fino ad allora sono permesse all'uscente solo le scritture
necessarie a completare questo perimetro organizzativo; dopo il merge
cessa nuove scritture comuni. Nessun successore e' gia' stato creato
o ha assunto il ruolo. Usare HANDOFF-001 in ORCHESTRATION.md.

## Perimetri approvati e stato

| Perimetro | Stato e responsabilita |
| --- | --- |
| CONT-001 rev.2 / ORCH-001 rev.2 | Protocollo di coordinamento; orchestratore titolare, con rilascio alla pubblicazione di questa consegna |
| MIG-FLOW-002 rev.3 | Collaudo delegato completo sufficiente; resta il nuovo OK per ogni scope successivo |
| MIG-016 | Chiusa con prova delegated MIG-016-delegated-001; nessun lavoro applicativo residuo approvato |
| Ambiente di collaudo dedicato | Creazione e collaudo autorizzati dall'utente; non autorizza altre copie/installazioni o trasferimenti dalle copie storiche |
| HANDOFF-001 / BRANCH-LIFECYCLE-001 | Pubblicazione organizzativa e pulizia remota autorizzate, con verifiche e conservazione della storia non integrata |
| MIG-017 | Proposta di fixture multidimensionali, NON approvata; nessun codice o incarico parallelo autorizzato |

Le approvazioni sopra sopravvivono al cambio chat, nei loro limiti.
Il successore non deve richiedere di nuovo approvazioni valide, ma non
puo' ampliare il perimetro o dedurre nuovi scope dal passaggio.

## Revisioni, PR e verifiche

- Ultimo main prima di questo lavoro: c1d3572627b0f544ae6ef54d05daca7ee2b62f62, chiusura MIG-016
  tramite PR #103, CI PR e post-merge positive.
- Commit effettivamente collaudato: dbdef7ab000d99d2e3dde68b0857840f0047d5fe.
  13 fasi Ruth, quattro fixture sintetiche passed, 11 test mirati OK.
  mode: delegated; nessuna esecuzione personale attribuita all'utente.
- La destinazione dedicata era stata aggiornata ai soli documenti di
  chiusura fino a c1d3572627b0f544ae6ef54d05daca7ee2b62f62, con working tree pulito.
  Questo non prova il suo stato nella nuova chat o l'allineamento alla
  presente pubblicazione: ricontrollare accesso, HEAD, permessi e ambiente.
- Nessun modello reale, peso, CUDA o inferenza verificato. Log, percorsi,
  dettagli dell'host ed evidenze grezze restano locali secondo DATA-001.
- UI del progetto: testo preparato in PROJECT_SETUP.md; inserimento
  e collaudo indipendente della continuita' restano pending.
- La verifica della presente pubblicazione richiede PR merged, CI success
  e lettura del ref main finale. Lo SHA della PR puo' essere diverso
  dal source_main_sha: ricostruire il diff, non assumere un codice diverso.

## Incarichi, branch, prerequisiti e blocchi

| ID | Responsabile dopo il rilascio | Stato | Prerequisiti / prossima azione |
| --- | --- | --- | --- |
| ORCH-TAKEOVER-001 | Successore designato dall'utente | pending | Risolvere main, leggere consegna e protocolli, verificare strumenti/accessi/permessi/ambiente e registrare claim esplicito |
| BRANCH-CLEANUP-001 | Nuovo orchestratore dopo claim | blocked | Servono create-tag e delete-ref autenticati; riconfermare live ogni ramo e incarico prima di agire |
| UI-CONT-001 | Utente per inserimento UI; orchestratore per verifica | pending | Usare il blocco PROJECT_SETUP.md, poi prova di ripresa senza modifiche applicative |
| MODEL-001 | Responsabile Models da riconfermare | active nei metadati | Descrive catalogo gia' integrato; non e' nuovo scope. Chiarire utilizzatori prima di cancellare rami del nodo |

Audit alla base: 97 branch, 85 contained_candidate e 11 archive_required,
nessuna PR/issue aperta e nessun tag. Nessuna cancellazione o creazione
tag eseguita. La presente PR aggiunge il ramo temporaneo sopra, che
resta pending_cleanup dopo merge se le mutazioni continuano a mancare.
Inventario e SHA in BRANCH_AUDIT.md; piano tag in BRANCH_ARCHIVE.md.
Non assumere assenza di nuovi branch/PR dopo l'audit.

Il connettore GitHub verificato permette commit/branch/PR/merge, ma non
espone create-tag/delete-ref. gh segnala autenticazione non valida.
Il controllo app list_projects fallisce con Codex app tool request failed:
nessuna creazione automatica della chat successore e' dimostrata.
Questi sono blocchi della sessione uscente, da riverificare nel successore;
non sono autorizzazioni a modificare login, recuperare token o attivare plugin.

Le risorse locali vanno indicate privatamente dall'utente. Non pubblicare
percorsi personali in questa consegna. Non cercare o modificare i checkout
storici. Le prove dell'uscente non trasferiscono permessi al successore.

## Presa in carico richiesta

Il successore prepara una PR limitata al claim, riportando:
handoff_id, proprio ID organizzativo, precedente titolare, SHA pubblicazione
letta, verifiche strumenti/locali eseguite e blocchi residui.
Prima del merge conferma che questa consegna non sia gia' claimed e che
non esista una contesa. Dopo il merge del claim e' l'unico titolare delle
scritture comuni. Nessun'altra modifica prima di questa assunzione.
La nuova chat resta Work Cloud fuori dal progetto; le chat branch ordinarie
restano aperte e fatte proseguire dall'utente.

Prossima azione: l'utente apre il successore con SUCCESSOR_MESSAGE.md;
il successore verifica e assume il ruolo, poi riprende solo gli incarichi
autorizzati e possibili. MIG-017 resta in attesa del nuovo OK.
