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

## GGUF-FIXTURE-MANUAL-001 accettata; MIG-012 chiusa (17 ottobre 2026)

- L'utente ha fornito la prova personale del commit main `a3ced17b0e4bccaf8fe8e10db9c1f5b5c508bda3`, verificato
  identico al ref GitHub heads/main prima della chiusura.
- L'ingresso Ruth senza argomenti ha completato tutte le dodici fasi in ordine,
  fino a FIXTURE, con riepilogo conclusivo regolare e nessun errore.
- FIXTURE: **passed** su tre file GGUF sintetici; CAT=ready, GGUF=inspected,
  META=inspected, INDEX=partial e BLOCKS=partial attesi per i tipi non
  verificati dalle fasi legacy, KBLOCKS=inspected.
- Suite mirata `test_pipeline`: **4 test, OK**, senza failure.
  I controlli CAT/GGUF/META/INDEX/BLOCKS/KBLOCKS sull'eventuale catalogo
  personale assente erano `not_configured`, comportamento valido.
- Sotto MIG-FLOW-002 revisione 2 la prova completa costituisce accettazione
  personale sufficiente: nessuna seconda autorizzazione richiesta.
  MIG-012: implementation **complete**, verification **passed** (CI e prova
  sintetica), integration **merged** (PR #71-#74), manual acceptance **passed**.
- Nessun GGUF reale o payload di tensori ispezionato nel collaudo personale;
  nessuna validazione di modelli, inferenza, runtime o CUDA.
- DATA-001: non pubblicare output grezzi, path, inventari hardware,
  identificativi host, versioni di pacchetti o benchmark. Solo commit,
  esiti sintetici e numero test. `local_sync: verified_on_tested_revision`
  per lo SHA sopra, secondo prova utente; dopo documenti di chiusura
  `local_sync: pending_docs_fast_forward`.
- Nessuna MIG-013 approvata: nuovo perimetro funzionale richiede verde
  esplicito ai sensi di MIG-FLOW-001. Le note storiche `pending`
  non descrivono lo stato attuale.

## MIG-013 / GGUF-STRUCT-SUMMARY-SCOPE-001 approvata (17 ottobre 2026)

- Approvazione esplicita dell'utente; base main `2a2ea80eee7b35c9c2d195e6c0de66ac57ee6550`.
- Contratto read-only: riepilogo per soli GGUF dichiarati in CAT dei lettori
  GGUF, META, INDEX, BLOCKS e KBLOCKS; stati `structural_checked`,
  `size_unverified`, `invalid`, `not_configured` (skip valido).
  L'esito INDEX/BLOCKS `size_unverified` puo' essere superato solo dalla
  verifica completa degli ingombri operata da KBLOCKS; gli sconosciuti
  rimangono `size_unverified`, errori strutturali diventano `invalid`.
- Tredicesimo passaggio del default Ruth `SUMMARY`, dopo FIXTURE;
  `gguf-summary` per ispezione mirata. Fixture sintetiche e test di regressione,
  CI e promozioni GGUF -> Formats -> Models -> main richiesti.
- Stato **implementazione candidata**, verifica **pending**, integrazione
  **pending**, `GGUF-STRUCT-SUMMARY-MANUAL-001` **pending** dopo merge.
  Collegamento locale: `local_sync: pending_no_access`.
- Nessuna lettura di pesi, download, rete, checksum payload, inferenza, CUDA
  o benchmark. DATA-001, REMOTE-001, MIG-FLOW-001/002 invariati.
  Collaudo personale positivo sufficiente senza seconda conferma.

## MIG-013 integrazione tecnica completata (17 ottobre 2026)

- PR #78 GGUF, #79 Formats, #80 Models e #81 main integrate; commit applicativo
  `6da03a348407c9caec355238c21c6a122a4b6298`.
- CI di tutte le promozioni **success**: unit test sintetici per file validi,
  quantizzazioni standard e K, tipo sconosciuto, header non valido,
  estensione incompleta, catalogo assente/non valido, CLI e regressioni;
  mappe ricorsive coerenti.
- Ruth default ora esegue **13 fasi**: SUMMARY dopo FIXTURE.
  Comando mirato `gguf-summary`; catalogo opzionale assente =
  `not_configured` non bloccante; anomalie = `invalid` oppure
  `size_unverified`, non certificate come successo.
- Stato: implementation **complete**; verification **passed (CI sintetica)**;
  integration **merged**; **GGUF-STRUCT-SUMMARY-MANUAL-001 pending**.
  Dopo aggiornamento conservativo del checkout personale, eseguire Ruth
  senza argomenti e `python -m unittest models.formats.gguf.tests.test_structural_summary -v`.
  Output completo positivo equivale ad accettazione senza secondo via libera
  (MIG-FLOW-002 rev.2).
- Non sono stati verificati GGUF reali, pesi, integrita' del payload,
  CUDA, runtime o inferenza. DATA-001/REMOTE-001 invariati;
  `local_sync: pending_no_access`. MIG-014 non approvata.

## GGUF-STRUCT-SUMMARY-MANUAL-001 accettata; MIG-013 chiusa (17 ottobre 2026)

- Commit main dell'esecuzione personale `49d3c451b0da86acc99ce33ec6c395f0c27e1ec7`, corrispondente al
  ref GitHub heads/main verificato prima della chiusura.
- Default Ruth: **13 controlli completati** in ordine, inclusi FIXTURE e
  SUMMARY, senza errori. Fixture sintetiche: **passed**, tre file.
- SUMMARY: `not_configured` con `CAT:not_configured` e catalogo personale
  opzionale assente; skip valido, non equivale alla verifica di GGUF reali.
- Suite mirata `test_structural_summary`: **11 test, OK**. Inclusi casi
  sintetici di formati supportati e sconosciuti e input strutturali errati.
- Prova positiva sufficiente come accettazione personale (MIG-FLOW-002
  revisione 2): non occorre una seconda conferma. MIG-013: implementation
  **complete**, verification **passed** (CI + collaudo sintetico personale),
  integration **merged** (PR #78-#81), manual acceptance **passed**.
- Nessun GGUF reale, pesi o integrita' payload verificati; nessuna
  dequantizzazione, installazione, download, CUDA, backend o inferenza.
- DATA-001: pubblicati soltanto SHA, esiti e numero test, non log,
  inventari host, percorsi, credenziali, risultati benchmark o dati reali.
- Checkout locale: `local_sync: verified_on_tested_revision` dalla prova
  personale sul commit indicato; dopo la PR documentale di chiusura,
  `local_sync: pending_docs_fast_forward` finche' non vi e' nuova prova.
- Le precedenti note candidate / manual pending sono **storiche**.
  MIG-014 non approvata: per ogni nuovo scope serve approvazione
  esplicita (MIG-FLOW-001).

## MIG-014 / GGUF-STRUCT-STABILITY-001 approvata (17 ottobre 2026)

- Utente ha approvato il perimetro proposto; commit base main
  `5a0a912d47e5860c8dcb8fd0188b6b24913314b1`.
- Estensione limitata alla fase SUMMARY già presente: osservazione dei
  metadati del file GGUF dichiarato prima/dopo lettori bounded.
  Identità (dispositivo/inode), dimensione, mode/link e timestamps ns;
  se cambiano: `invalid` / `STABILITY:changed_during_inspection`;
  se non verificabili o divenuti indisponibili: `invalid` /
  `STABILITY:unavailable`. I report privi di differenze osservate
  conservano lo stato originale, anche `size_unverified`.
- Solo test sintetici per file stabile, sostituzioni, variazioni metadati,
  rimozione e symlink. 13 fasi Ruth invariate; nessun payload letto.
- Non è una garanzia atomica di immutabilità o hash/integrità crittografica.
  Non copre modifiche non rilevate dai due snapshot. DATA-001 e
  REMOTE-001 invariati, Kaggle inattivo.
- Stato attuale: scope **approved**, implementazione **candidate**,
  verifica **pending**, integrazione **pending**, collaudo personale
  `GGUF-STRUCT-STABILITY-MANUAL-001` **pending** dopo merge.
  Branch `models/formats/gguf/work/GGUF-STRUCT-STABILITY-001`.
  Local checkout non accessibile: `local_sync: pending_no_access`.

## MIG-014 integrazione tecnica completata (17 ottobre 2026)

- PR #84 (GGUF), #85 (Formats), #86 (Models), #87 (main) integrate con
  GitHub CI **success** su ogni promozione (test sintetici, regressioni,
  punto d'ingresso e mappe). Commit applicativo `1a77914e1e3ed9ee7cdbe7a57c2ddccffce7b616`.
- Il controllo file stabile è in SUMMARY, **13 fasi Ruth invariate**.
  Confronta attributi e identità osservabili prima e dopo la sequenza di
  lettori già autorizzati; file cambiato -> `invalid` /
  `STABILITY:changed_during_inspection`, file divenuto indisponibile ->
  `invalid` / `STABILITY:unavailable`. Un file invariato mantiene i
  risultati precedenti anche `size_unverified`.
- Suite `test_stability.py`: **9 test sintetici** passati in CI;
  simulati mutazione dimensionale, sostituzione stesso ingombro, cambi
  timestamp, cancellazione, symlink, casi stabili validi e invalidi.
- Stato: implementation **complete**, verification **passed (CI)**,
  integration **merged**; **GGUF-STRUCT-STABILITY-MANUAL-001 pending**.
  Prova personale post-merge: default Ruth 13 fasi e 9 test mirati `test_stability`
  su commit main verificato. L'evidenza positiva equivale ad accettazione
  senza doppia conferma (MIG-FLOW-002 rev.2).
- Nessun peso o modello reale verificato, nessun hash payload; il confronto
  non garantisce immutabilità atomica né integrità crittografica. Nessuna
  lettura tensor_data, rete, download, CUDA, inferenza o benchmark.
  DATA-001 e REMOTE-001 invariati; `local_sync: pending_no_access`.
  Nessuna nuova migrazione approvata.

## GGUF-STRUCT-STABILITY-MANUAL-001 accettata; MIG-014 chiusa (17 ottobre 2026)

- Prova personale ricevuta sull'esatta revisione main `b1662aab99cbc97b6203dc41e7355521e702a2cd`,
  verificata identica a GitHub heads/main prima della chiusura.
- `main.py` senza argomenti: **13 fasi completate** in ordine senza errori;
  fixture GGUF sintetiche: **passed**, tre file. SUMMARY:
  `not_configured` / `CAT:not_configured` con catalogo opzionale assente,
  skip valido; non e' stata effettuata ispezione di un GGUF reale.
- `models.formats.gguf.tests.test_stability`: **9 test, OK**,
  zero fallimenti, sui controlli sintetici di mutazione/stabilita'.
- La prova positiva e' accettazione sufficiente in MIG-FLOW-002 rev.2,
  **senza ulteriore consenso**. MIG-014: implementation **complete**,
  verification **passed** (CI + collaudo personale), integration
  **merged** (PR #84-#87), personal acceptance **passed**.
- Non sono stati certificati pesi, hash, immutabilita' atomica, integrita',
  modelli reali, CUDA, backend o inferenza: snapshot best-effort soltanto.
- DATA-001: registrati solo SHA, stati e conteggi; nessun log grezzo,
  inventario host, percorso personale, credenziale o benchmark in Git.
- Revisione locale riferita dal collaudo utente: `local_sync: verified_on_tested_revision`.
  Dopo il merge documentale e' necessario il fast-forward dei soli documenti:
  `local_sync: pending_docs_fast_forward` sino a nuova prova; non si presume
  verificato uno stato Git pulito del checkout personale.
- MIG-015 non approvata: presentare perimetro funzionale distinto per
  autorizzazione ai sensi di MIG-FLOW-001. Le precedenti note `pending`
  sono cronologia storica, non stato corrente.
