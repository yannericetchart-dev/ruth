# Ruth: punto di ingresso comune

Repository canonico: <https://github.com/yannericetchart-dev/ruth>.
Branch di ingresso: `main`. Protocollo: `CONT-001`, revisione 1.

Questo indice e' comune a ChatGPT e Codex. Leggerlo nuovamente da GitHub
all'inizio di ogni attivita', fissando la revisione letta. Le copie locali e
le fonti caricate possono essere precedenti: non usarle come fonte di
modifiche pubblicate. In REMOTE-001 lo sviluppo versionato e' svolto tramite
GitHub e la sincronizzazione del checkout locale avviene solo dopo il merge
in main. Il clone principale non va modificato durante lo sviluppo.
L'indice non implica che il codice dei branch di sviluppo sia gia' in main.

## Letture di ripresa

1. [Istruzioni operative](AGENTS.md).
2. [Decisioni vigenti](DECISIONS.md).
3. [Checkpoint corrente](CHECKPOINT.md).
4. [Procedura di configurazione e collaudo](PROJECT_SETUP.md).
5. [Roadmap](ROADMAP.md), [codemap](CODEMAP.md), [apimap](APIMAP.md)
   e mappe del componente e dei suoi antenati.
6. [Regole dello scaffold](ARCHITECTURE.md) e checkpoint del componente.

## Stato del progetto

La riorganizzazione e' stata autorizzata per passaggi: prima la continuita'
fra i client, poi lo scaffold documentale, infine la migrazione incrementale.
L'architettura proposta resta evolvibile. Il checkpoint indica branch,
revisioni e stato di integrazione di ciascuna porzione. Un'implementazione
nel branch di lavoro non deve essere considerata gia disponibile in main.
I branch storici rimangono fonti conservate della migrazione.

Kaggle e' un'integrazione futura inattiva. Log, misure, modelli e dati di
macchina rimangono locali secondo DATA-001. Non sono fonti pubbliche di stato.

## Risposta iniziale attesa

Prima di sviluppare, indicare obiettivo e prossimo passo dal checkpoint,
decisioni rilevanti, revisione consultata e eventuali divergenze locali.
Per il lavoro corrente applicare REMOTE-001, che sostituisce SYNC-001:
nessun blocco di ripresa dovuto al mancato allineamento del clone locale.
Se manca una fonte o un collaudo non e' stato svolto, dichiararlo. Non
dedurre il completamento di una fase dalla sola presenza dei documenti.

## Regola del punto d'ingresso (MAIN-FLOW-001)

Avviando `main.py` senza argomenti Ruth esegue automaticamente **tutte le
operazioni integrate e approvate** in sequenza secondo le loro dipendenze.
I sottocomandi restano disponibili per verifiche mirate, non sono il flusso
normale. Flusso integrato corrente: PY-001, DEP-001, DIAG-001 e HW-001 (CPU/RAM), con il
solo controllo minimo di `pip`; l'inventario completo dei requisiti non e'
ancora definito. DIAG usa report preesistenti, senza modifiche host. MAIN-FLOW-001 e' integrata in main tramite PR #17 e verificata da CI.
Il collaudo personale dell'avvio sequenziale senza argomenti e' stato
superato sul commit `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`
(MAIN-MANUAL-001 complete). Consultare CHECKPOINT.md per il prossimo
perimetro, che richiede una nuova approvazione funzionale.

## Stato DIAG-001

DIAG-001 e' integrata tecnicamente in main tramite PR #22 e verificata
su GitHub. Il collaudo locale DIAG-MANUAL-001 e' stato confermato positivo sulla
revisione `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`, con ordine PY -> DEP -> DIAG.
Consultare CHECKPOINT.md per il prossimo perimetro da approvare.

## HW-SCOPE-001 approvato

Perimetro iniziale hardware CPU/RAM read-only autorizzato il
16 ottobre 2026; lo stato verificato, le promozioni e il successivo
collaudo personale HW-MANUAL-001 sono nel CHECKPOINT.md.

## HW-001 integrata tecnicamente (16 ottobre 2026)

Inventario CPU/RAM tramite `hardware.detect()` e quarto passaggio del
flusso `main.py`, dopo PY->DEP->DIAG. Verifiche sintetiche CI superate,
PR #27 integrata in main. HW-MANUAL-001 superato personalmente
sulla revisione `59b26fd6b427e6756ce0285fee60590cb077ebf1` il 16 ottobre 2026.
Prossima migrazione soltanto dopo nuovo perimetro approvato;
consultare CHECKPOINT.md.

## GPU-SCOPE-001 approvato

Il 16 ottobre 2026 e' stato approvato solo il provider NVIDIA con
interrogazione read-only a `nvidia-smi`. AMD/Intel non autorizzati;
lo stato di verifica, integrazione e collaudo e' in CHECKPOINT.md.

## GPU-001 NVIDIA integrata tecnicamente (16 ottobre 2026)

Il quinto passaggio di `main.py` interroga soltanto NVIDIA in sola
lettura; verifica CI superata e PR #33 integrata in main.
GPU-MANUAL-001 e' stata superata personalmente il 16 ottobre 2026
sulla revisione `c344c71013cfa05da4f49f3204900f4bdc89b8ea`; tutte le cinque fasi completate.
La successiva migrazione richiede nuova approvazione del perimetro
(MIG-FLOW-001). AMD/Intel e CUDA non sono migrati.

## CAT-SCOPE-001 approvato

Avvio della porzione Models catalogo locale, approvata il
16 ottobre 2026: manifest JSON opzionale ignorato da Git,
sesto passaggio Ruth; nessuna scansione/download/GGUF/inferenza.
Stato implementazione, CI e collaudo CAT-MANUAL-001 in CHECKPOINT.md.

## CAT-001 integrata tecnicamente (16 ottobre 2026)

`main.py` include come sesto passaggio il catalogo modelli locale
`.local/models/catalog.json` opzionale; manifest assente o vuoto
sono stati validi e non causano fallimento. PR #38 in main e CI
superate; **CAT-MANUAL-001 e' stata superata personalmente**
il 16 ottobre 2026 sul commit `90e57f106fb664b0257775260044125ea85228e2`.
Il prossimo dominio richiede un nuovo perimetro funzionale esplicito,
ancora non approvato (MIG-FLOW-001).

## GGUF-SCOPE-001 approvato (16 ottobre 2026)

L'utente ha approvato GGUF-001, sola lettura dell'intestazione dei file
GGUF dichiarati in CAT, come settima fase Ruth. Le PR #41-#44 hanno
integrato il codice in main, commit `15b106082c4dee37622c61a8afefd248a57464a0`, dopo CI positiva.
**GGUF-MANUAL-001 superato personalmente il 16 ottobre 2026**
sul commit `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`.
Con catalogo assente il settimo controllo e' stato correttamente
`not_configured`; non e' stata eseguita una lettura di GGUF reale.
Nuova migrazione solo dopo perimetro esplicito approvato.
Nessun download, metadato esteso, peso tensor o inferenza autorizzato.

## GGUF-META-SCOPE-001 approvato (16 ottobre 2026)

L'utente ha approvato una lettura read-only, bounded, dei metadati
tecnici GGUF dichiarati in CAT, come ottava fase dopo GGUF.
Le PR #47, #48, #49 e #50 hanno integrato il componente in main
(commit `a630c83f6c62a229e18fe5e64f0863c19de16018`), con CI e test sintetici superati.
**GGUF-META-MANUAL-001 superato personalmente il 16 ottobre 2026**
con il flusso completo di Ruth sul commit `d318fd793da514f21ee2e8475304d1a6c883ee94`.
Catalogo e metadati GGUF `not_configured`: stato valido; nessun
file GGUF reale provato. Consultare CHECKPOINT.md.
Nessun download, lettura pesi o inferenza autorizzato.

## GGUF-INDEX-SCOPE-001 approvato (16 ottobre 2026)

L'utente approva MIG-009: verifica bounded degli indici tensori dei soli
GGUF catalogati, senza leggere pesi. Nono passaggio automatico Ruth.
Implementazione candidata; verifiche e promozioni pendenti: consultare
CHECKPOINT.md. Collaudo GGUF-INDEX-MANUAL-001 solo dopo main.

## GGUF-INDEX-001 integrata tecnicamente (16 ottobre 2026)

Le PR #53 (gguf), #54 (formats), #55 (Models) e #56 (main) hanno
integrato la nona fase Ruth, INDEX, sul commit `1d51cfeed2bec83119ebf56ff0cb3840089484cf`, con CI
sintetica e mappe verificate. Solo indici descrittori, mai tensor_data.
**GGUF-INDEX-MANUAL-001 resta in attesa di collaudo personale**.
Nessuna migrazione successiva e' autorizzata.

## GGUF-INDEX-MANUAL-001 superato personalmente (17 ottobre 2026)

L'utente ha confermato positivo il collaudo del flusso completo Ruth
a nove fasi sul main `09045243e944514d40b9e58c6159b013efa2a196`. Catalogo e INDEX
`not_configured`: risultato previsto. Nessun GGUF reale o payload
di tensori verificato manualmente. MIG-009 complete. I precedenti
stati pending sono storici; prossimo scope ancora da approvare.

## GGUF-BLOCK-SCOPE-001 approvato (17 ottobre 2026)

MIG-010 approvata esplicitamente: cinque tipi quantizzati GGML
(Q4_0, Q4_1, Q5_0, Q5_1 e Q8_0) per verifica bounded degli
ingombri dei tensori GGUF dichiarati, senza leggere i pesi.
Decima fase Ruth dopo INDEX, scope in
`models/formats/gguf/BLOCK_SCOPE.md`. Codice candidato:
CI e promozioni ancora da verificare. Collaudo personale
GGUF-BLOCK-MANUAL-001 richiesto dopo il merge in main.

## GGUF-BLOCK-001 integrata tecnicamente (17 ottobre 2026)

Le PR #59 (GGUF), #60 (Formats), #61 (Models) e #62 (main)
integrano il controllo BLOCKS, decima fase del default Ruth.
Commit applicativo `5f896f1bbacaa8c802f66f2a40ee7663cfa40604`, verifiche CI sintetiche e mappe
superate. Solo metadati, descriptor ed estensione byte: nessun peso
del modello letto. **GGUF-BLOCK-MANUAL-001 resta pending** fino al
collaudo personale sul main aggiornato e conferma esplicita.
Le precedenti note candidate sono storiche.

## GGUF-BLOCK-MANUAL-001 superato personalmente (17 ottobre 2026)

L'utente conferma l'esito positivo del flusso Ruth a dieci fasi
sul commit main `2f42a9d7a631e8779613784b58f8ce1b92afff9c`. Il catalogo locale non configurato
e gli skip di GGUF, META, INDEX e BLOCKS sono validi;
nessun file GGUF reale testato. MIG-010 complete; prossimo
perimetro ancora da proporre e approvare (MIG-FLOW-001).
Le precedenti note di manual pending sono storiche.

## GGUF-KBLOCK-SCOPE-001 approvato (17 ottobre 2026)

Autorizzata MIG-011: calcolo strutturale bounded degli ingombri
dei sei tipi K-quant GGML Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K, soltanto
per GGUF del catalogo. Undicesimo passaggio Ruth KBLOCKS dopo
BLOCKS. Scope: `models/formats/gguf/KBLOCK_SCOPE.md`.
Codice candidato sul ramo remoto; CI/promozioni in attesa.
GGUF-KBLOCK-MANUAL-001 richiesta dopo main; nessun peso letto.

## GGUF-KBLOCK-001 integrata tecnicamente (17 ottobre 2026)

Le PR #65 GGUF, #66 Formats, #67 Models e #68 main hanno
integrato l'undicesima fase `KBLOCKS` in Ruth.
Commit applicativo `a803e15e872d71a96b7b2de37b13a8964d37d168`, test sintetici e mappe
CI superati. Il validatore legge solo descrittori dei sei
tipi K-quant supportati e non i byte dei tensori.
**GGUF-KBLOCK-MANUAL-001 pending** dopo merge documentale:
prova personale di Ruth e fixture sintetiche prima di
qualunque nuova migrazione. Le note candidate sono storiche.

## MIG-FLOW-002 revisione 2 e GGUF-KBLOCK-MANUAL-001 (17 ottobre 2026)

L'utente dispone che, se incolla un log di collaudo personale completo,
positivo e riferito alla revisione main verificata, esso costituisca
accettazione sufficiente: non chiedere un secondo consenso. Completare
autonomamente checkpoint, mappe, PR/CI/merge e proporre lo scope successivo,
senza scriverne il codice prima del nuovo verde (MIG-FLOW-001).
Questo prevale sulle istruzioni storiche di conferma separata.

Prova MIG-011: flusso Ruth 11 fasi e 9 test sintetici K-block
superati (`OK`) sul commit `233efc358f56a66d2abf796fa8e20a476aa788cf`.
GGUF-KBLOCK-MANUAL-001 e MIG-011 complete dopo la chiusura
documentale. Catalogo `not_configured` valido; nessun GGUF reale
verificato. Leggere l'ultimo CHECKPOINT per lo SHA di chiusura.

## MIG-012 (17 ottobre 2026)

GGUF-FIXTURE-SCOPE-001 approvata: collaudo sintetico CAT->GGUF->META->INDEX->BLOCKS->KBLOCKS, dodicesimo passaggio FIXTURE. Stato candidato, non ancora collaudato dall'utente. Nessun modello reale o peso letto. Consultare CHECKPOINT.md.

## MIG-012 integrazione tecnica verificata (17 ottobre 2026)

- PR #71 (GGUF), #72 (Formats), #73 (Models) e #74 (main) integrate, commit applicativo main `b057957b5d96b3c09ecde0b6c28ffc0e977c6942`.
- CI per tutte e quattro le promozioni success, inclusi test sintetici, punto d'ingresso e mappe. Dodicesima fase Ruth FIXTURE, dopo KBLOCKS, e comando mirato `gguf-fixture`.
- Stato: implementation complete, verification passed (fixture e CI), integration merged. **GGUF-FIXTURE-MANUAL-001 pending**: richiede collaudo personale del default Ruth e test mirati dopo aggiornamento del checkout personale.
- Non sono stati testati modelli reali o letti pesi. Nessun download, rete, CUDA, backend o inferenza. Nuovo perimetro bloccato fino a collaudo personale e nuova approvazione (MIG-FLOW-001/002). DATA-001 e REMOTE-001 invariati.
