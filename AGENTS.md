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

## Avvio sequenziale Ruth (MAIN-FLOW-001)

Decisione accettata dall'utente il 16 ottobre 2026. `main.py` **senza
argomenti** e' l'orchestratore ordinario: deve eseguire automaticamente,
una dopo l'altra e nell'ordine delle dipendenze, **tutte le funzionalita'
integrate e autorizzate**, con output chiaro per ogni passaggio e riepilogo.
Non limitare l'avvio normale all'ultima funzione migrata e non richiedere
all'utente sottocomandi manuali per eseguire il flusso completo. I comandi
espliciti sono soltanto strumenti di verifica mirata.

Ogni nuova porzione approvata e integrata in main deve aggiungere il proprio
passaggio all'orchestrazione automatica (MIG-FLOW-002), aggiornando i test
di ordine, blocchi, esito e il checkpoint. Se un prerequisito fallisce,
non eseguire operazioni dipendenti: indicare che sono state saltate e perche'.
Non inventare requisiti di pacchetti, configurazioni, dati o effetti non
approvati. Stato attuale: PY-001, DEP-001, DIAG-001, HW-001 (CPU/RAM) e GPU-001 (NVIDIA) in sequenza; DEP esegue
solo verifica minima di `pip`, **non** elenco completo delle dipendenze Ruth;
nessuna installazione o modifica dell'host (DATA-001).

## Diagnostica read-only dei report (DIAG-SCOPE-001)

Perimetro DIAG-001 approvato il 16 ottobre 2026: PY e DEP continuano
in sequenza; DIAG li segue e trasforma i loro report gia' disponibili
in motivi e suggerimenti, senza leggere host/log o modificare .venv.
Se PY fallisce, DEP e' saltato ma DIAG deve comunque spiegare l'errore.
I passaggi futuri non devono sostituire quelli precedenti nel default.
Non equiparare assenza di segnalazioni a compatibilita' nativa certificata.

## HW-SCOPE-001: inventario hardware iniziale

Approvata il 16 ottobre 2026 la prima porzione read-only CPU e RAM.
L'ingresso normale `main.py` deve aggiungere HW dopo PY->DEP->DIAG,
con messaggi anche se un prerequisito dell'ambiente non e' conforme;
HW non dipende dalla .venv valida per osservare CPU/RAM. Non dedurre
presenza GPU, CUDA, backend di inferenza, RAM disponibile o core fisici
perche' la CPU o la RAM totale sono osservabili. Dati reali soltanto
locali (DATA-001). Provider NVIDIA/AMD/Intel in componenti futuri.

## GPU-SCOPE-001: solo provider NVIDIA

Perimetro GPU-SCOPE-001 approvato 16 ottobre 2026: aggiungere
unicamente l'inventario NVIDIA read-only da nvidia-smi, come quinto
passaggio indipendente della CLI Ruth. AMD e Intel restano inattive,
e GPU rilevata non equivale a compatibilita CUDA/driver/backend.
Non riportare inventari reali, stderr o identificativi persistenti
in Git (DATA-001). Dopo promozione serve GPU-MANUAL-001 (MIG-FLOW-002).

## CAT-SCOPE-001: primo modello dichiarativo locale

Approvato il 16 ottobre 2026: solo manifest JSON opzionale in
`.local/models/catalog.json` con riferimenti relativi confinati, report
read-only e sesta fase automatica di main.py; `catalog` mirato.
No scansione, no GGUF binary, no Hugging Face, no installazione,
no download o inferenza. Nessun dato modelli reale su Git (DATA-001).
Dopo merge serve collaudo personale CAT-MANUAL-001 (MIG-FLOW-002).

## CAT-001: integrazione tecnica completata

CI sintetica e promozioni fino a main effettuate tramite PR #36,
#37 e #38 (16 ottobre 2026). Default `main.py` ora esegue
PY->DEP->DIAG->HW->NVIDIA->CAT; nessuna config catalogo e' un esito
valido. CAT-MANUAL-001 e' ancora in attesa: non procedere con
GGUF, download o altri componenti prima del collaudo dell'utente.

## GGUF-META-001: stato corrente (16 ottobre 2026)

GGUF-META-SCOPE-001 e' implementato, verificato su GitHub e integrato
tramite PR #47-#50, main `a630c83f6c62a229e18fe5e64f0863c19de16018`. MAIN-FLOW-001
ora include l'ottavo controllo di metadati GGUF dichiarati.
L'utente deve ancora eseguire personalmente il nuovo main ed
accettare GGUF-META-MANUAL-001 (MIG-FLOW-002). Le precedenti
indicazioni CAT/GGUF pending in questo file sono storiche e
non prevalgono sul checkpoint piu' recente. Nessuna nuova migrazione
autorizzata; DATA-001 e REMOTE-001 restano invariati.

## GGUF-META-MANUAL-001: accettazione (16 ottobre 2026)

- L'utente conferma esplicitamente positivo il collaudo personale
  del flusso ordinario di Ruth a otto fasi sul main `d318fd793da514f21ee2e8475304d1a6c883ee94`.
- Catalogo, GGUF e META `not_configured`: skip valido.
  Nessuna prova manuale con modello reale ne' certificazione inferenza.
- GGUF-META-001: implementata, CI verificata, integrata (PR #50),
  manual acceptance **passed/complete**. Le note pending precedenti
  restano storiche, non rappresentano lo stato corrente.
- Prossima porzione senza approvazione: richiede nuovo scope
  secondo MIG-FLOW-001/002. DATA-001 e REMOTE-001 invariati.

## GGUF-INDEX-001: stato corrente (16 ottobre 2026)

GGUF-INDEX-SCOPE-001 approvato; implementato, CI superata e integrato
tramite PR #53-#56 fino a main `1d51cfeed2bec83119ebf56ff0cb3840089484cf`.
La nona fase INDEX e' nel flusso default di Ruth (MAIN-FLOW-001).
**GGUF-INDEX-MANUAL-001 resta pending** fino a prova personale e
accettazione esplicita dell'utente dopo la sincronizzazione.
I precedenti stati "candidate" sono storici. Nessuna nuova porzione
e' approvata; REMOTE-001, DATA-001 e MIG-FLOW-001/002 invariati.

## GGUF-INDEX-MANUAL-001: collaudo accettato (17 ottobre 2026)

- GGUF-INDEX-MANUAL-001: **complete/passed**, accettazione esplicita
  dell'utente il 17 ottobre 2026 dopo esecuzione personale del flusso
  ordinario `.venv/bin/python main.py` sul commit `09045243e944514d40b9e58c6159b013efa2a196`,
  identico a main verificato su GitHub.
- Nove fasi in ordine PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT ->
  GGUF -> META -> INDEX; riepilogo finale di completamento regolare.
- Catalogo e letture GGUF/META/INDEX `not_configured` sono skip validi
  e non bloccanti. Nessun file GGUF reale ispezionato personalmente;
  non risultano validati contenuti dei tensori, pesi o compatibilita'
  con CUDA/backend/inferenza.
- La prova manuale completa il gate MIG-FLOW-002 per MIG-009;
  una nuova migrazione richiede uno scope separato e approvazione
  esplicita (MIG-FLOW-001). Nessun nuovo componente avviato.
- DATA-001 e REMOTE-001 invariati: in Git solo esito e commit;
  niente path, nomi modelli, output grezzi, inventari, log o benchmark.
Le note precedenti che indicano GGUF-INDEX-MANUAL-001 pending non
prevalgono sul checkpoint aggiornato.

## GGUF-BLOCK-001: integrazione tecnica (17 ottobre 2026)

GGUF-BLOCK-SCOPE-001 implementato, CI verificata, integrato con
PR #59-#62 fino al main `5f896f1bbacaa8c802f66f2a40ee7663cfa40604`; Ruth default ora ha dieci
fasi e BLOCKS segue INDEX. GGUF-BLOCK-MANUAL-001 **pending**:
non iniziare nuova migrazione prima dell'accettazione personale e
di un ulteriore scope funzionale esplicitamente approvato.
Vincoli REMOTE-001, DATA-001 e MIG-FLOW-001/002 invariati.
Le annotazioni precedenti di candidato sono storiche.

## GGUF-BLOCK-MANUAL-001: accettazione positiva (17 ottobre 2026)

- L'utente ha eseguito personalmente il flusso di dieci controlli
  di `.venv/bin/python main.py` sulla revisione `2f42a9d7a631e8779613784b58f8ce1b92afff9c`,
  coincidente con il main pubblicato; ha confermato esplicitamente
  positivo il collaudo GGUF-BLOCK-MANUAL-001 il 17 ottobre 2026.
- Tutte le fasi PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF ->
  META -> INDEX -> BLOCKS sono state completate in ordine.
  Il catalogo opzionale assente e' risultato `not_configured`
  e GGUF/META/INDEX/BLOCKS sono skip validi e non bloccanti.
- Non e' stato collaudato personalmente alcun file GGUF reale:
  l'esito non certifica pesi, integrita', dequantizzazione,
  CUDA, backend, runtime o inferenza.
- I risultati di CI e la verifica con fixture sintetiche dei cinque
  tipi Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 restano distinti
  dall'accettazione manuale senza catalogo.
- DATA-001 e REMOTE-001 rispettati: non vengono versionati
  inventari dell'host, percorsi reali, log, manifest, modelli,
  benchmark o credenziali. Nessuna nuova migrazione approvata.
MIG-010 e il gate manuale sono complete. Le note anteriori
che indicano `GGUF-BLOCK-MANUAL-001 pending` restano storiche e
non rappresentano la fase corrente.

## GGUF-KBLOCK-001: stato tecnico corrente (17 ottobre 2026)

GGUF-KBLOCK-SCOPE-001 e' implementato, verificato con fixture
sintetiche e integrato tramite PR #65-#68, main `a803e15e872d71a96b7b2de37b13a8964d37d168`.
MAIN-FLOW-001 ora include undici fasi, KBLOCKS dopo BLOCKS.
GGUF-KBLOCK-MANUAL-001 resta **pending**: esecuzione personale
del default Ruth e test locali sintetici dopo merge main,
seguiti da conferma esplicita dell'utente (MIG-FLOW-002).
Le precedenti note di codice candidato sono storiche.
Non avviare MIG-012 prima della conferma e nuovo scope approvato.
REMOTE-001 e DATA-001 restano vigenti.
