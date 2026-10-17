# Ruth: checkpoint di progetto

Revisione documentale: 52 (indice main); CONT-001 revisione 2.
Fase 3: migrazione incrementale per componente.
Stato: **MIG-016 GGUF-MULTITENSOR-FIXTURE-001 implementata, verificata in CI e integrata in main; GGUF-MULTITENSOR-MANUAL-001 passed tramite collaudo delegato MIG-FLOW-002 rev.3; MIG-016 chiusa**.
MIG-015 e GGUF-FIXTURE-SUMMARY-MANUAL-001 restano complete.
MIG-014 e GGUF-STRUCT-STABILITY-MANUAL-001 restano complete.
MIG-013 e GGUF-STRUCT-SUMMARY-MANUAL-001 restano complete.
MIG-012 e GGUF-FIXTURE-MANUAL-001 restano complete.
GGUF-KBLOCK-MANUAL-001 e MIG-011 restano complete.
GGUF-BLOCK-MANUAL-001 resta complete.
GGUF-INDEX-MANUAL-001 rimane complete.
GGUF-MANUAL-001 e CAT-MANUAL-001 restano complete.
PY, DEP, DIAG, HW/CPU e GPU NVIDIA restano accettate personalmente.
L'ingresso senza argomenti di Ruth esegue tutte le funzioni integrate.

## Protocollo corrente: CONT-001 rev.2 / ORCH-001 / MIG-FLOW-002 rev.3

- Approvazione esplicita dell'utente il 17 ottobre 2026.
- Cambiamento solo documentale: ruoli, incarichi, parallelismo approvato,
  prerequisiti, integrazioni seriali e collaudo delegato sufficiente alla
  chiusura quando tutte le verifiche prescritte sono soddisfatte.
- Responsabile unico dei documenti comuni e dell'integrazione in main:
  orchestratore. Nessun incarico parallelo o nuovo perimetro funzionale
  viene avviato con questa pubblicazione.
- Sviluppo GitHub-first; DATA-001 invariata. Decisioni sostituite conservate
  come storiche in DECISIONS.md; gli obblighi personali rev.2 nelle note
  applicative datate non prevalgono sulla revisione 3.
- Stato documentazione: implementata, verificata e integrata tramite
  PR #101, main 5dbd00f1ee8adbf7a093ccfb0d5b96e738d25234.
  CI PR 39034668530 e CI main 39034711512: success, incluse mappe,
  regressioni sintetiche e controllo dell'assenza di modifiche generate.
  Albero integrato identico al candidato revisionato; solo dieci documenti
  e metadati di protocollo, nessun file applicativo/test/workflow modificato.
  Durante quella pubblicazione non erano stati eseguiti Ruth o test locali.
  Il collaudo delegato successivo e la chiusura MIG-016 sono registrati
  separatamente sotto, senza attribuire la prova al commit della chiusura.
- Istruzioni UI: blocco aggiornato preparato in PROJECT_SETUP.md;
  inserimento UI e collaudo indipendente della continuita' rev.2 pending.
  La pubblicazione GitHub non cambia automaticamente la UI.
- Copia pulita precedente, non piu' destinazione del collaudo delegato:
  stato storico mantenuto separato. Verificata in lettura: remoto canonico, main,
  nessuna modifica tracciata o non tracciata, commit
  31730f6d4f712adb0cad8715ae166567f3926e77, .venv presente.
  Permessi OS di scrittura non disponibili su checkout, metadati Git e
  .venv; percorso fuori dalle radici di scrittura autorizzate.
  previous_clean_checkout: pending_permissions. Non sono stati provati aggiornamenti.
- Copia storica modificata preservata integralmente ed esclusa dalla sync.
  Nessuna importazione automatica dalla copia storica. Una nuova destinazione
  dedicata e' stata autorizzata separatamente e verificata come indicato sotto.
- GGUF-MULTITENSOR-MANUAL-001 conserva l'ID storico del gate ed e'
  soddisfatta con mode: delegated, status: passed. Evidenza
  MIG-016-delegated-001 sul commit dbdef7ab000d99d2e3dde68b0857840f0047d5fe:
  default Ruth 13 fasi, quattro fixture sintetiche passed e 11 test mirati OK.
  Valutazione umana aggiuntiva non prescritta. Nessuna esecuzione personale
  attribuita all'utente.
- Nuova destinazione dedicata designata e autorizzata dall'utente:
  validation_checkout: verified_on_tested_revision. Prove di accesso,
  revisione, ambiente e pulizia superate. Nessun trasferimento da copie
  o ambienti esistenti; nessuna installazione ulteriore.
- Prossima azione locale: dopo merge dei soli documenti di chiusura,
  sincronizzare conservativamente la nuova destinazione, verificando
  il diff documentale senza attribuire un nuovo collaudo non eseguito.
  Lo stato delle copie precedenti rimane distinto.
- MIG-017 non approvata. Dopo chiusura verificata di MIG-016 proporre lo
  scope successivo e attendere nuovo OK prima di implementarlo.

## Cronologia applicativa

Le registrazioni seguenti descrivono le revisioni e le prove dell'epoca.
Lo stato corrente e il gate vigente sono quelli riportati sopra.

## Collaudi confermati

- PY-001: ispezione Python/.venv, collaudo PY-MANUAL-001 positivo sul
  commit `ef4654af8ab106ef194af254d9f3884e5cabb6b0`.
- DEP-001: metadati dipendenze Python, collaudo DEP-MANUAL-001
  positivo sul commit `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- MAIN-FLOW-001: primo flusso sequenziale, collaudo MAIN-MANUAL-001
  sul commit `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`.
- DIAG-001: diagnostica dei report Python/dipendenze, collaudo
  DIAG-MANUAL-001 sul commit
  `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`.
- HW-001/CPU-001: CPU e RAM totale read-only, collaudo HW-MANUAL-001
  sul commit `59b26fd6b427e6756ce0285fee60590cb077ebf1`.
- GPU-001: perimetro NVIDIA approvato (GPU-SCOPE-001), provider
  read-only `nvidia-smi`. Promozioni PR #30 (NVIDIA), PR #31 (GPU),
  PR #32 (hardware), PR #33 (main, merge
  `54a02cf37262a78f5a1ed0aa84cd6d210060a14c`).
  Chiusura tecnica checkpoint/mappe tramite PR #34,
  merge `c344c71013cfa05da4f49f3204900f4bdc89b8ea`.
- CI GitHub runs 38918374871, 38918650589, 38918759116,
  38918867451 e 38919460341 **success**: 92 test sintetici
  e 258 mappe coerenti. La CI non usa GPU reali.
- **GPU-MANUAL-001: complete**. L'utente ha aggiornato il proprio
  checkout main tramite fast-forward e verificato HEAD
  `c344c71013cfa05da4f49f3204900f4bdc89b8ea`. Ha eseguito personalmente
  `.venv/bin/python main.py` senza argomenti il 16 ottobre 2026.
  Sono confermati, nell'ordine, PY -> DEP -> DIAG -> HW CPU/RAM ->
  NVIDIA, rilevamento NVIDIA con stato `available` e riepilogo
  finale positivo. Non riportare specifiche concrete della macchina
  o output grezzi nel repository (DATA-001).

## Limiti e protezioni

- Ruth continua a controllare automaticamente solo `pip` fra le
  dipendenze Python: la presenza non equivale a compatibilita'.
  I report `present_unverified` e `ok_unverified` non certificano
  la disponibilita' nativa, la sicurezza o l'inferenza.
- Il report GPU identifica solo dispositivi NVIDIA tramite
  `nvidia-smi` disponibile e interrogabile; non certifica
  driver/CUDA, VRAM libera, backend llama.cpp, prestazioni o
  inferenza. AMD e Intel GPU restano componenti futuri inattivi.
- Nessuna installazione, modifica host o benchmark eseguiti.
- DATA-001: credenziali, inventari reali, log completi, percorsi,
  specifiche di host e dati privati restano locali ed esclusi da Git;
  Kaggle inattivo.

## Sincronizzazione e continuita' (REMOTE-001)

- `local_sync: verified_on_tested_revision` per copia pulita main
  al commit `c344c71013cfa05da4f49f3204900f4bdc89b8ea`, sulla base del fast-forward e SHA
  comunicati dall'utente.
- I successivi commit **documentali** della presente chiusura
  richiederanno solo `git fetch origin` e un nuovo
  `git merge --ff-only origin/main` dopo verifica di checkout
  pulito (`local_sync: pending_docs_fast_forward`).
  Il collaudo precedente resta valido: nessun codice eseguibile
  cambiato in questa chiusura.
- Il vecchio checkout con modifiche locali rimane intatto
  (`legacy_checkout: blocked_divergence`).
  Non eseguire reset, clean, stash o rebase automatici.

## Prossima azione

MIG-006/CAT-001 e' completata e accettata personalmente sul commit
`90e57f106fb664b0257775260044125ea85228e2`. GGUF-SCOPE-001 e'
stato approvato, verificato, integrato e accettato personalmente sul commit
`ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`. Nessuna porzione successiva e' gia' autorizzata.
La prossima migrazione richiede **prima** un perimetro funzionale esplicito
(scopo, API, ingressi, effetti, controlli, limiti, test e passaggio
automatico `main.py`) con nuovo semaforo verde dell'utente
(MIG-FLOW-001). Il successivo collaudo personale resta obbligatorio
dopo ogni integrazione in main (MIG-FLOW-002).

## Decisioni vigenti

CONT-001, DATA-001, MIG-FLOW-001/002, REMOTE-001, MAIN-FLOW-001,
HW-SCOPE-001, GPU-SCOPE-001, PLAN-001, SCOPE-001, DEP-ORDER-001
e DIAG-SCOPE-001. SYNC-001 resta storica/superseded.

## CAT-001: scope approvato, lavoro candidato (16 ottobre 2026)

- CAT-SCOPE-001 approvato esplicitamente: catalogo dichiarativo da
  `.local/models/catalog.json` opzionale, verifiche sintattiche e
  di disponibilita' senza file discovery, download o GGUF binary.
- Candidato read-only `models/catalog`, parent `models`, sesto
  passaggio `main.py` (PY->DEP->DIAG->HW->NVIDIA->CAT) e sottocomando
  mirato `catalog`. Fixture locali temporanee per test.
- Stato: candidate implemented; verification not_run;
  integration not_merged; CAT-MANUAL-001 planned.
- DATA-001: nessun contenuto di manifest, percorsi o dati reali in Git.
  REMOTE-001: nessun checkout principale modificato.

## CAT-001: catalogo locale integrato, test personale pendente (16 ottobre 2026)

- CAT-SCOPE-001 approvato esplicitamente: soltanto lettura di
  `.local/models/catalog.json` opzionale, con riferimenti relativi e
  protezione da percorsi non consentiti, duplicati, symlink e JSON invalido.
- Contratti frozen `ModelRecord`, `ModelCatalog`, API
  `models.catalog.catalog` e `resolve`. Nessun download, scanner,
  decodifica GGUF, sorgente Hugging Face o inferenza.
- `main.py` esegue ora **PY -> DEP -> DIAG -> HW CPU/RAM -> NVIDIA -> CAT**;
  lo stato `not_configured` (manifest assente) e `empty` (nessun
  modello dichiarato) sono esiti validi e non bloccanti. A disposizione
  anche il sottocomando mirato `main.py catalog`.
- Promozioni: PR #36 in models/catalog/main (merge
  `a562e1b4ce948d3865a15a9d06454d91b8cb7cde`),
  PR #37 in models/main (merge `77e3a7327c9f0750d6e8363766000ba830ea1c5e`),
  PR #38 in main (merge `12c395ba3bb272d66095d4e468794936b1693892`).
- GitHub Actions 38928909391, 38929028829, 38929118836: **success**;
  116 test sintetici (21 CAT, 20 root/CLI) e 264 mappe coerenti.
- Stati CAT: implementation complete, verification passed,
  integration merged. **CAT-MANUAL-001 active**; non avviare GGUF,
  acquisizione o altre migrazioni prima della prova utente.
- `local_sync: pending_no_access` dopo merge Models; la copia pulita
  deve essere aggiornata solo con fast-forward sicuro e senza perdere
  modifiche. `legacy_checkout: blocked_divergence` invariato.
- DATA-001: non versionare manifest o elenco reale di modelli,
  percorsi locali e output utente. Kaggle resta inattivo.

## Chiusura CAT-MANUAL-001 (16 ottobre 2026)

- Conferma esplicita dell'utente: collaudo personale **positivo** con
  `.venv/bin/python main.py` senza argomenti sul checkout main al commit
  `90e57f106fb664b0257775260044125ea85228e2` (SHA locale dichiarato e coincidente con GitHub).
- Sequenza di sei fasi confermata: PY -> DEP (solo pip) -> DIAG -> HW CPU/RAM
  -> NVIDIA -> CAT; riepilogo conclusivo regolare. L'assenza del manifest
  catalogo produce `not_configured`, stato valido e non bloccante.
- Stato conclusivo CAT-001: implementation complete; verification passed
  (CI sintetica gia' registrata); integration merged (PR #38);
  manual acceptance **passed** (CAT-MANUAL-001).
- Le precedenti note che indicano CAT-MANUAL-001 active/pending/blocked
  restano registrazioni storiche, sostituite dalla presente chiusura.
- `local_sync: verified_on_tested_revision` per HEAD `90e57f106fb664b0257775260044125ea85228e2`
  comunicato dall'utente; dopo questo aggiornamento documentale in main
  `local_sync: pending_docs_fast_forward` finche' non sara' sincronizzato.
  Vecchio checkout divergente non toccato.
- Nessun dato reale, inventario, output grezzo o manifest e' pubblicato
  (DATA-001). La prossima porzione richiede nuovo scope e via libera esplicito
  (MIG-FLOW-001), prima di qualsiasi migrazione.

## GGUF-SCOPE-001 autorizzato (16 ottobre 2026)

- GGUF-MANUAL-001 non avviato: e' necessario solo dopo l'integrazione.
- Nuova porzione MIG-007 autorizzata esplicitamente, dopo il collaudo CAT.
  Scopo: introspezione read-only dei soli 24 byte di header GGUF per modelli
  dichiarati da CAT; controlli di magic, versione, endian e contatori.
- Candidato remoto su `models/formats/gguf/work/GGUF-001`;
  implementazione candidata, verifiche/CI e promozioni ancora pendenti.
- Nessun download, metadata payload, tensor, inferenza, CUDA o modifiche host.
  Nessun dato reale in Git (DATA-001).

## GGUF-001 integrata tecnicamente (16 ottobre 2026)

- GGUF-SCOPE-001 approvato esplicitamente dopo CAT-MANUAL-001.
- Lettura bounded di 24 byte dall'intestazione GGUF, solo per i modelli
  dichiarati e validati nel manifest CAT; report immutabili, versioni
  2/3 e byte order di v3, gestione codici errore. Non si controllano
  metadati completi, pesi o compatibilita' per inferenza.
- Sequenza ufficiale Ruth `main.py` senza argomenti:
  PY -> DEP (solo pip) -> DIAG -> HW CPU/RAM -> NVIDIA -> CAT -> GGUF.
  Catalogo assente, vuoto o senza GGUF: skip valido; il controllo GGUF
  e' read-only e non sostituisce nessuna fase precedente.
- Promozioni e CI superate: PR #41 `models/formats/gguf/main`
  (merge `f783a4521f785e69374516b02c3079e99f678c78`);
  PR #42 `models/formats/main` (`f1730969b7b024f2cdf528d64ee7e87dca566a66`);
  PR #43 `models/main` (`4c2ece369738092af546eae0aa2967db59490008`);
  PR #44 `main` (`15b106082c4dee37622c61a8afefd248a57464a0`).
- CI runs 38935705385, 38935861881, 38935943414, 38936032188:
  tutte **success**, incluse verifiche sintetiche della CLI e mappe;
  nessuna prova su modelli reali in CI.
- Stati: implementation complete, verification passed (CI sintetica),
  integration merged. **GGUF-MANUAL-001 active/pending**:
  l'utente deve sincronizzare in sicurezza il checkout pulito e avviare
  `.venv/bin/python main.py` per confermare l'esito della settima fase.
- `local_sync: pending_no_access` per la nuova revisione main; il clone
  legacy divergente resta intatto. L'ultimo collaudo personale CAT
  resta valido e non viene ripetuto per i documenti.
- Nessun dato reale di host o manifest, log, modello o benchmark in Git.
  GGUF metadata payload, Hugging Face, CUDA e inferenza non approvati.
  Prossima porzione soltanto dopo GGUF-MANUAL-001 positivo e nuovo
  perimetro esplicitamente approvato (MIG-FLOW-001/002).

## Chiusura GGUF-MANUAL-001 (16 ottobre 2026)

- L'utente ha comunicato il fast-forward fino a main `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`,
  il controllo del commit HEAD e l'esecuzione personale di
  `.venv/bin/python main.py` senza argomenti; ha poi confermato
  esplicitamente **esito positivo**.
- Ordine verificato: PY -> DEP (pip minimo) -> DIAG -> HW CPU/RAM ->
  NVIDIA -> CAT -> GGUF; completamento regolare.
  Catalogo e GGUF `not_configured` sono stati validi.
- Accettazione limitata al caso di catalogo assente: non dimostra
  l'ispezione di un file GGUF reale, la verifica dei pesi,
  la compatibilita' con CUDA o la disponibilita' di inferenza.
- Stati: implementation complete; verification passed (CI sintetica);
  integration merged (PR #44); **GGUF-MANUAL-001 passed/complete**.
  Le precedenti note pending sono storiche e sostituite da questa voce.
- `local_sync: verified_on_tested_revision` per `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`
  comunicato dall'utente; `local_sync: pending_docs_fast_forward`
  per questa successiva modifica di sola documentazione.
  Il checkout legacy resta intatto.
- DATA-001: esclusi dal Git inventari reali, log, output grezzi,
  manifest e modelli; registrati solo esito e commit.
- Nessuna successiva migrazione e' approvata. Occorre nuovo perimetro
  e semaforo verde (MIG-FLOW-001).

## GGUF-META-SCOPE-001 approvato (16 ottobre 2026)

- Autorizzazione esplicita dell'utente al perimetro di sola lettura dei
  metadati tecnici GGUF; base main `044f0504b8e023689c0b78bc981440fbc06aee12`, dopo
  GGUF-MANUAL-001 positivo. Dettaglio in
  `models/formats/gguf/METADATA_SCOPE.md`.
- Attivita MIG-008: candidato in sviluppo su
  `models/formats/gguf/work/GGUF-META-001`; non e' ancora
  verificato ne' integrato in main.
- Nuove API dichiarative architettura / file_type / context_length,
  con valori assenti sconosciuti e parsing bounded di strutture KV.
- Ottava fase default PY -> DEP -> DIAG -> HW -> NVIDIA ->
  CAT -> GGUF -> GGUF metadata; catalogo assente non blocca.
- Nessuna lettura tensor, nessun download, installazione,
  compatibilita' inferenza, dato locale Git o nuova migrazione.
- Prossimo passo: test sintetici, mappe, CI e promozioni fino
  a main; poi collaudo personale GGUF-META-MANUAL-001.

## GGUF-META-001 integrata tecnicamente (16 ottobre 2026)

- Perimetro approvato GGUF-META-SCOPE-001 invariato: solo
  architettura, codice file_type e context_length dichiarati,
  da GGUF elencati nel catalogo CAT-001, senza scansioni.
- Lettore bounded in `models/formats/gguf/runtime/metadata.py`
  con report immutabili e controlli su versioni v2/v3, endian,
  lunghezze, array, profondita' e file malformati. Non legge tensori.
- L'ingresso Ruth `main.py` default ora avvia tutte le otto fasi:
  PY -> DEP (pip minimo) -> DIAG -> HW CPU/RAM -> NVIDIA ->
  CAT -> GGUF -> GGUF metadata. `main.py gguf-metadata` resta
  verifica mirata. Un catalogo non configurato o vuoto non blocca.
- Promozioni complete: PR #47 models/formats/gguf/main,
  #48 models/formats/main, #49 models/main, #50 main
  (merge `a630c83f6c62a229e18fe5e64f0863c19de16018`).
- CI runs 38940033886, 38940265778, 38940360999,
  38940625555: **success**, inclusi test sintetici del parser,
  test della CLI e controllo delle mappe. Nessun modello reale in CI.
- Stato: implementation complete (scope bounded), verification passed
  (CI sintetica), integration merged; **GGUF-META-MANUAL-001 pending**.
  Le precedenti voci "candidate" sono storiche e non guidano il lavoro.
- `local_sync: pending_no_access` per il nuovo main; dopo
  l'aggiornamento documentale verra' comunicato lo SHA finale.
  Checkout legacy non modificato (REMOTE-001).
- Nessun dato di host, manifest, log o modello pubblicato (DATA-001);
  rimangono esclusi tensor payload, download, llama.cpp, CUDA, inference.
  Nessuna porzione seguente e' gia' approvata; attesa accettazione
  manuale personale prima di presentare il nuovo perimetro.

## GGUF-META-MANUAL-001: accettazione personale (16 ottobre 2026)

- L'utente ha eseguito il fast-forward non distruttivo di main,
  verificato `git rev-parse HEAD` e avviato personalmente
  `.venv/bin/python main.py` sul commit `d318fd793da514f21ee2e8475304d1a6c883ee94`,
  identico al main pubblicato su GitHub.
- Conferma esplicita dell'utente: **collaudo positivo**.
- Tutte le otto fasi PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT ->
  GGUF -> META hanno prodotto il riepilogo conclusivo atteso.
- Catalogo, intestazioni GGUF e metadati GGUF `not_configured`
  sono skip validi. Non e' stata effettuata prova manuale su
  GGUF reale ne' certificazione di pesi, CUDA o inferenza.
- GGUF-META-001: implementation complete; verification passed
  (CI sintetica); integration merged (PR #50);
  **manual acceptance passed** (GGUF-META-MANUAL-001).
- Le note anteriori pending/blocked/candidate sono storiche,
  sostituite da questo checkpoint.
- `local_sync: verified_on_tested_revision` per `d318fd793da514f21ee2e8475304d1a6c883ee94`
  comunicato dall'utente; dopo la pubblicazione documentale
  `local_sync: pending_docs_fast_forward`. Nessun checkout modificato.
- DATA-001: nessun output grezzo, inventario, path, manifest,
  modello o log reale pubblicato. Prossima porzione ancora
  non approvata: necessario perimetro e via libera MIG-FLOW-001.

## GGUF-INDEX-SCOPE-001 approvato (16 ottobre 2026)

- MIG-009 approvata esplicitamente dopo la chiusura di MIG-008;
  base main `240e1bc8616f9087390c08603704db4165ea1587`, ramo di lavoro remoto
  `models/formats/gguf/work/GGUF-INDEX-001`.
- Scopo: lettura limitata e read-only dei soli descrittori tensore
  dichiarati in GGUF del catalogo, senza leggere il tensor_data.
- Verifica offset, nomi duplicati, dimensioni, allineamento e
  intervalli di tipi scalari conosciuti. Quantizzazioni/tipi
  sconosciuti esplicitamente size_unverified, mai success implicito.
- Nona fase Ruth dopo GGUF META e comando mirato `gguf-index`;
  catalogo assente non bloccante. Fixture sintetiche e mappe candidate.
- Stato **candidate implemented**, verification **not_run**,
  integration **not_merged** fino a CI e promozioni; nessun
  collaudo GGUF-INDEX-MANUAL-001 ancora eseguibile.
- Nessuna modifica all'host, download, tensor payload, CUDA o
  inferenza; dati reali restano locali (DATA-001).

## GGUF-INDEX-001 integrata tecnicamente (16 ottobre 2026)

- Autorizzazione funzionale GGUF-INDEX-SCOPE-001 registrata; soli
  descrittori tensore di GGUF dichiarati in CAT e dati tensor esclusi.
- Implementazione `models/formats/gguf/runtime/tensor_index.py`:
  v2/v3 ed endian, parser KV limitato, max 16384 descrittori,
  verifica nomi, dimensioni, offset relativi e allineamento.
  Intervalli certi solo per tipi scalari con ampiezza nota;
  sconosciuti e quantizzati producono `size_unverified` e stato partial.
- Nono passaggio dell'ingresso ufficiale `main.py` senza argomenti:
  PY -> DEP (pip minimo) -> DIAG -> HW -> NVIDIA -> CAT ->
  GGUF -> META -> INDEX. Sottocomando mirato `gguf-index`.
  Catalogo assente/vuoto senza GGUF = skip valido e non bloccante.
- PR #53 foglia gguf/main (merge 8b608074ef08bbae41b8d68565e9fc38ea66e248),
  #54 formats/main (405c77d1dbfeb0e1ffc66bb0ea261494a9f3546f),
  #55 models/main (90b7fc5a025279c24a678d52def31b8f37576b61),
  #56 main (`1d51cfeed2bec83119ebf56ff0cb3840089484cf`); CI success runs 38945221302,
  38945375637, 38945463381, 38945568401.
- Stato: implementation complete per scope autorizzato,
  verification passed (solo fixture sintetiche),
  integration merged. **GGUF-INDEX-MANUAL-001 active/pending**:
  avvio personale di Ruth sul nuovo main, conferma esplicita richiesta.
- `local_sync: pending_no_access` per il nuovo main; il clone
  principale non e' stato modificato. Il collaudo positivo precedente
  di GGUF-META resta valido, senza certificazione di modelli reali.
- Nessun peso, modello, inventario, benchmark, percorso, manifest
  o log reale in Git (DATA-001). Kaggle inattivo. CUDA, Hugging Face,
  download e inferenza fuori scope. Nuova migrazione bloccata
  fino al collaudo personale e a successivo scope autorizzato.

## Chiusura GGUF-INDEX-MANUAL-001 (17 ottobre 2026)

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
- Stato finale: implementation complete; verification passed
  (test e CI sintetici); integration merged (PR #56);
  **GGUF-INDEX-MANUAL-001 complete** e **MIG-009 complete**.
- `local_sync: verified_on_tested_revision` per `09045243e944514d40b9e58c6159b013efa2a196`,
  come comunicato dall'utente; successivo commit solo documentale:
  `local_sync: pending_docs_fast_forward`. Il checkout originale
  non e' stato modificato dall'assistente.
- Le note precedenti candidate/active/pending/blocked sono storiche
  e non guidano la fase corrente. Il prossimo passo e' definire e
  approvare un nuovo perimetro senza anticipare implementazioni.

## GGUF-BLOCK-SCOPE-001 approvata; MIG-010 candidata (17 ottobre 2026)

- Via libera esplicito per cinque tipi quantizzati GGML;
  base main `6f472b0ac182b113c5c727c3f8f5c291c074de2e`, ramo remoto
  `models/formats/gguf/work/GGUF-BLOCK-001`.
- Aggiunta candidata al reader INDEX con calcolo opzionale delle
  dimensioni Q4_0/Q4_1/Q5_0/Q5_1/Q8_0; controllo sui soli
  descrittori e dimensioni del file. Nessun payload letto.
- API immutabili inspect_block_layout e inspect_all_block_layouts.
  Flusso futuro di dieci fasi con BLOCKS dopo INDEX e
  sottocomando mirato `gguf-blocks`.
- Stato della nuova porzione **candidate implementation**;
  verification **not_run**, integration **not_merged**.
  Le precedenti nove fasi restano implementate, verificate,
  integrate e accettate personalmente.
- Collaudo GGUF-BLOCK-MANUAL-001 pianificato soltanto dopo
  promozioni e CI positive. Nessun nuovo perimetro approvato.
  Local sync `pending_no_access`; REMOTE-001 e DATA-001 invariati.

## GGUF-BLOCK-001: integrazione tecnica completata (17 ottobre 2026)

- Perimetro GGUF-BLOCK-SCOPE-001 approvato, codice in
  `models/formats/gguf/runtime/blocks.py` e estensione
  opt-in di `tensor_index.py` per Q4_0, Q4_1, Q5_0,
  Q5_1 e Q8_0 (32 elementi per blocco, 18/20/22/24/34 byte).
- Catalog-only, sola lettura bounded dei descrittori, verifica
  prima dimensione divisibile per 32, calcolo ingombri,
  bound degli offset e sovrapposizione. Gli altri formati
  rimangono `size_unverified`, senza falso successo.
- Ruth con dieci passi: PY -> DEP -> DIAG -> HW -> NVIDIA ->
  CAT -> GGUF -> META -> INDEX -> BLOCKS. Sottocomando
  `gguf-blocks`. Catalogo non configurato non bloccante.
- Promozioni CI superate: PR #59 gguf/main (merge
  e6236ce45623bad4d4e9df2f80289a59fb30c11d),
  #60 formats/main (d456646059b5eafe4364b265b8fe774fe8269103),
  #61 models/main (3172c609440bfa0849df0b908e54a3d91ae27bce),
  #62 main (`5f896f1bbacaa8c802f66f2a40ee7663cfa40604`). Runs 39022476943,
  39022574327, 39022637716, 39022698747: **success**,
  incluso `tools/maps.py --check`.
- Stato scope: implementation **complete**,
  verification **passed** (solo fixture sintetiche),
  integration **merged**. **GGUF-BLOCK-MANUAL-001 pending**:
  collaudo personale sulla revisione main aggiornata necessario.
- `local_sync: pending_no_access`: checkout personale non
  modificato da remoto; i dati reali di host/modelli, log,
  benchmark e credenziali restano esclusi da Git (DATA-001).
- Nessun payload letto, hash, dequantizzazione, CUDA,
  inference, download o acquisizione. Kaggle inattivo.
  Le note candidate precedenti sono storiche; la prossima
  migrazione e' bloccata fino al collaudo personale e
  ad approvazione di uno scope distinto.

## GGUF-BLOCK-MANUAL-001: collaudo accettato (17 ottobre 2026)

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
- Stato definitivo dello scope: implementation **complete**,
  verification **passed** (CI con fixture sintetiche),
  integration **merged** (PR #62), manual
  acceptance **passed/complete** (GGUF-BLOCK-MANUAL-001).
  Root MIG-010 **complete**, senza blocchi aperti.
- `local_sync: verified_on_tested_revision` per `2f42a9d7a631e8779613784b58f8ce1b92afff9c`,
  come comunicato dall'utente; dopo merge di questi soli documenti,
  `local_sync: pending_docs_fast_forward`. Il clone principale
  non e' stato modificato dall'assistente.
- Le annotazioni anteriori candidate/active/pending/blocked sono
  storiche: questo checkpoint ne sostituisce lo stato.
  Passo successivo: definire un nuovo scope separato e richiedere
  approvazione esplicita prima di modificare altro codice.

## GGUF-KBLOCK-SCOPE-001 approvato, MIG-011 candidata (17 ottobre 2026)

- Via libera esplicito ai sei K-quant GGML da 256 elementi;
  base main `1b497172bfad562e6096e88ac8be69ef01a8ced6`, ramo remoto
  `models/formats/gguf/work/GGUF-KBLOCK-001`.
- API e fixture candidate per GGUF-KBLOCK-001, read-only
  e catalog-bound: Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K.
  La semantica di INDEX e BLOCKS rimane invariata.
- Ruth default prevista con 11 fasi, KBLOCKS dopo BLOCKS;
  comando mirato `gguf-kblocks`. Catalogo assente skip valido.
- Stato nuova porzione: candidate implementation,
  verification not_run, integration pending; collaudo
  GGUF-KBLOCK-MANUAL-001 pianificato dopo main.
- Nessun tensor_data letto, rete/download, CUDA, benchmark,
  inferenza, dato locale in Git (DATA-001). Checkout principale
  intatto (REMOTE-001). Prossimo scope ancora da autorizzare.

## GGUF-KBLOCK-001: integrazione tecnica completata (17 ottobre 2026)

- Scope GGUF-KBLOCK-SCOPE-001 approvato e implementato in
  `models/formats/gguf/runtime/kblocks.py`, estensione
  opt-in di `tensor_index.py`; sei GGML Q2_K/Q3_K/Q4_K/
  Q5_K/Q6_K/Q8_K, blocchi 256:84/110/144/176/210/292.
- Il parser e' CAT-only, bounded e read-only; controlla
  divisibilita' prima dimensione, ingombri, allineamento,
  bounds e sovrapposizione. INDEX/BLOCKS non cambiati;
  tipi non previsti restano `size_unverified`.
- MAIN-FLOW-001: undici passi PY -> DEP -> DIAG -> HW ->
  NVIDIA -> CAT -> GGUF -> META -> INDEX -> BLOCKS -> KBLOCKS.
  Comando mirato `main.py gguf-kblocks`; catalogo assente
  o senza GGUF e' skip valido senza download.
- Promozioni: PR #65 leaf GGUF, #66 Formats, #67 Models,
  #68 main `a803e15e872d71a96b7b2de37b13a8964d37d168`. CI runs 39025028855,
  39025109274, 39025159357, 39025210189: **success**,
  fixture v2/v3/endian e regressioni, `tools/maps.py --check`.
- Stato tecnico: implementation **complete**,
  verification **passed** (solo fixture sintetiche),
  integration **merged**. **GGUF-KBLOCK-MANUAL-001
  active/pending**: avvio personale di Ruth dalla revisione
  main aggiornata e suite fixture sintetiche locale;
  accettazione esplicita ancora mancante.
- `local_sync: pending_no_access` per il nuovo main.
  Il checkout locale dell'utente non e' stato modificato.
  GGUF reale e compatibilita' inferenza non verificati.
- Nessun payload, modello reale, manifest, log, benchmark,
  credenziale o inventario pubblicato (DATA-001). Kaggle
  inattivo; no CUDA, backend, download, inferenza.
  Prossima migrazione bloccata fino a collaudo personale
  e separata approvazione di un nuovo scope.

## GGUF-KBLOCK-MANUAL-001 accettata; MIG-FLOW-002 revisione 2 (17 ottobre 2026)

- Prova comunicata dall'utente il 17 ottobre 2026 per main
  `233efc358f56a66d2abf796fa8e20a476aa788cf`, risolto e confrontato al ref heads/main GitHub:
  flusso normale `.venv/bin/python main.py` concluso senza errori
  su undici fasi in ordine, inclusa KBLOCKS.
- CAT, GGUF, META, INDEX, BLOCKS, KBLOCKS: `not_configured`
  come skip validi con catalogo opzionale assente.
- Suite mirata `test_kblocks.py`: **9 test superati, esito OK**,
  senza failure; verifica manuale sintetica distinta dalla
  verifica precedente CI e dall'uso di un GGUF reale.
- Evidenza positiva sufficiente come accettazione personale ai sensi
  della nuova MIG-FLOW-002 revisione 2; l'utente ha autorizzato
  espressamente la chiusura automatica senza una seconda conferma.
- Nessun GGUF reale, dato tensore, dequantizzazione, CUDA,
  backend, download o inferenza collaudati. Nessun dato host,
  path, output grezzo o log pubblicato (DATA-001).
- Stato finale MIG-011: implementation **complete**;
  verification **passed** (CI e fixture sintetiche);
  integration **merged** (PR #68); manual acceptance **passed**
  (GGUF-KBLOCK-MANUAL-001). ROOT MIG-011 **complete**,
  `blocked_by=[]`. Non vi sono migrazioni aggiuntive autorizzate.
- `local_sync: verified_on_tested_revision` per
  `233efc358f56a66d2abf796fa8e20a476aa788cf` sulla base della prova utente;
  `local_sync: pending_docs_fast_forward` per il successivo
  commit documentale main. Il clone locale non e' stato modificato
  dall'assistente.
- Regola stabile MIG-FLOW-002-R2: quando l'utente incolla prova
  sufficiente di SHA main, flusso integrato e test specifici
  passati, registrare accettazione direttamente, senza un
  secondo "si"; con evidenze non sufficienti o errori non chiudere.
  Dopo merge proporre lo scope successivo, da approvare con
  MIG-FLOW-001. Le precedenti note manual pending sono storiche.

## MIG-012 / GGUF-FIXTURE-001 approvata (17 ottobre 2026)

- Approvazione esplicita utente. Base main `518ef0d2bdeeafe281a52cde192438fb96eb3f40`, branch `models/formats/gguf/work/GGUF-FIXTURE-001`.
- Scope: tre file GGUF sintetici in directory temporanea, verifica CAT/GGUF/META/INDEX/BLOCKS/KBLOCKS, dodicesimo passaggio Ruth FIXTURE.
- Stati: implementation candidate, verification pending, integration pending, personal acceptance pending.
- Nessun dato di host o modello reale pubblicato; solo fixture generate a runtime. Prossima verifica: CI, mappe, PR e promozione; poi collaudo personale GGUF-FIXTURE-MANUAL-001.

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

## MIG-015 / GGUF-FIXTURE-SUMMARY-001 approvata (17 ottobre 2026)

- Utente ha approvato esplicitamente la porzione MIG-015 dopo il collaudo MIG-014.
  Commit base main `0cb12ad90e2ef7d1b3b3b77009f9cf476a3e5289`.
- La fase FIXTURE legge anche SUMMARY sul catalogo *solo temporaneo* dei tre
  GGUF sintetici. Richiesti: `SUMMARY=structural_checked`, tre modelli con
  `structural_checked`, conteggi e ragioni esatti; esito anomalo ->
  `failed` con motivazione portabile. Gli stati INDEX/BLOCKS parziali legacy
  restano legittimi quando KBLOCKS verifica i formati noti.
- Nuova suite mirata `test_fixture_summary` (sette test sintetici) con
  formati sconosciuti, magic invalida, truncation, mutazioni e fallimenti
  indotti di SUMMARY. Regressione `test_pipeline` allineata.
- Ruth conserva **13 fasi**, la fase SUMMARY finale sul catalogo reale resta
  separata dalla SUMMARY temporanea interna a FIXTURE.
- Stato iniziale: scope **approved**, implementazione **candidate**,
  verification **pending**, integration **pending**, manual gate
  `GGUF-FIXTURE-SUMMARY-MANUAL-001` **pending** post-merge.
- Nessun modello reale, payload, download, rete, benchmark, CUDA o inferenza;
  DATA-001, REMOTE-001, MIG-FLOW-001/002 vigenti; Kaggle inattivo.

## MIG-015 integrazione tecnica completata (17 ottobre 2026)

- PR #90 GGUF, #91 Formats, #92 Models, #93 main: integrate dopo GitHub
  CI **success** in ogni promozione (test sintetici, regressione `test_pipeline`,
  punto d'ingresso e mappe). Commit applicativo main `91ad6f4e927c0ed2dd96b147a44c074ccbc5bfe4`.
- La dodicesima fase FIXTURE include ora il controllo SUMMARY sullo stesso
  catalogo temporaneo di **3 GGUF sintetici**, richiedendo
  `SUMMARY=structural_checked` e tre rapporti `structural_checked`.
  Il report deve fallire se SUMMARY è anomala. La SUMMARY standard finale
  resta la **tredicesima fase Ruth** sul catalogo locale opzionale, invariata.
- Sette test mirati `test_fixture_summary` superati in CI; coperti
  casi stabili, quantizzazioni note, tipo ignoto, magic non valida,
  extent troncato, modifiche durante la lettura e failure guard.
- Stato: implementazione **complete**, verification **passed (CI)**,
  integrazione **merged**; collaudo personale
  `GGUF-FIXTURE-SUMMARY-MANUAL-001` **pending**.
  Dopo il merge del checkpoint: eseguire Ruth 13 fasi e la suite mirata
  `python -m unittest models.formats.gguf.tests.test_fixture_summary -v`
  (sette test). Evidenza positiva completa vale come accettazione senza
  secondo consenso ai sensi di MIG-FLOW-002 rev.2.
- Non verificati GGUF reali o pesi; nessuna lettura tensor_data, hash
  payload, download, rete, CUDA o inferenza. DATA-001 e REMOTE-001 vigenti.
  `local_sync: pending_no_access`. Nuova migrazione non autorizzata.

## GGUF-FIXTURE-SUMMARY-MANUAL-001 accettata; MIG-015 chiusa (17 ottobre 2026)

- Utente ha fornito prova personale sul commit main `86d7aa5b2000e07ef55798cab3779bf53e3cbf72`,
  identico al ref GitHub `heads/main` verificato prima della chiusura.
- Ruth `main.py` senza argomenti: **13 fasi completate** in ordine, senza
  anomalie; collaudo FIXTURE: **passed** su tre GGUF sintetici,
  `SUMMARY=structural_checked`; INDEX e BLOCKS `partial` previsti per
  quantizzazione legittima, KBLOCKS `inspected`.
- SUMMARY successiva sul catalogo locale personale opzionale:
  `not_configured` / `CAT:not_configured`; skip valido, non collaudo reale.
- Suite mirata `test_fixture_summary`: **7 test, OK**, nessun fallimento;
  comprende casi sintetici struttura valida, tipo sconosciuto, magic errata,
  intervallo troncato, mutazione osservabile e guard di fallimento SUMMARY.
- MIG-FLOW-002 revisione 2: prova positiva = accettazione personale senza
  nuovo consenso. MIG-015: implementation **complete**, verification
  **passed** (CI e test sintetici personali), integration **merged**
  (PR #90-#93), manual acceptance **passed**.
- Nessun GGUF reale, peso, tensor_data o compatibilita' runtime testati;
  nessun download, rete, CUDA o inferenza. DATA-001: nessun log o inventario
  di host, percorso privato, credenziale o benchmark scritto su Git.
- Checkout locale: `local_sync: verified_on_tested_revision` limitatamente
  allo SHA del collaudo fornito; dopo merge dei documenti di chiusura
  `local_sync: pending_docs_fast_forward`. Stato del checkout pulito non
  dimostrato dall'output incollato.
- Le note precedenti `manual pending` sono storiche, non stato corrente;
  MIG-016 **non approvata**. Nuova porzione solo dopo approvazione esplicita
  del perimetro proposto, secondo MIG-FLOW-001.

## MIG-016 / GGUF-MULTITENSOR-FIXTURE-001 approvata (17 ottobre 2026)

- Approvazione esplicita dell'utente, base commit main
  `3cd6cad3e431f5ad1ab8dc698aee96330ff79ed1`.
- FIXTURE conserva i tre GGUF sintetici MIG-015 e aggiunge un quarto file
  con **tre tensori** (scalare F32, Q4_0, Q4_K), offset distinti e dimensioni
  note. CAT/GGUF/META/INDEX/BLOCKS/KBLOCKS/SUMMARY vengono eseguiti con
  catalogo temporaneo, senza dati reali. Summary richiede
  `structural_checked` per tutti e quattro, `tensor_count=3` per multi.
- Suite `test_multitensor` (11 test sintetici) per layout valido, descrittori
  non ordinati, endian big, alignment custom, offset non valido e fuori file,
  overlap, truncation, duplicati, dimensione blocco errata e tipi ignoti
  (`size_unverified`). Test precedenti fixture 3->4 aggiornati.
- **13 fasi Ruth invariate**. Nessun nuovo parser, modello reale, lettura
  tensor data, hash, dequantizzazione, CUDA, download, rete o inferenza.
  DATA-001, REMOTE-001, MIG-FLOW-001/002 invariati.
- Stato: autorizzata; codice **candidate**, verifica **pending**,
  integrazione **pending**, collaudo personale
  `GGUF-MULTITENSOR-MANUAL-001` **pending** solo dopo main.
  Branch `models/formats/gguf/work/GGUF-MULTITENSOR-FIXTURE-001`.
  Local checkout non accessibile: `local_sync: pending_no_access`.

## MIG-016 integrazione tecnica completata (17 ottobre 2026)

- PR #96 GGUF, #97 Formats, #98 Models e #99 main integrate dopo
  GitHub CI **success** a ogni promozione (suite sintetiche, regressioni,
  punto d'ingresso e mappe); commit applicativo main `4eb92225e3addbc8b7d58b15243557dae86e4479`.
- FIXTURE ora dichiara **4 GGUF sintetici** in catalogo temporaneo:
  `scalar`, `q4`, `multi` e `kquant`. `multi` ha tre descrittori
  (F32, Q4_0, Q4_K), offset 0/32/64 e intervalli validi.
  CAT/GGUF/META/INDEX/BLOCKS/KBLOCKS/SUMMARY sempre in sequenza;
  INDEX/BLOCKS possono ancora produrre `partial` per tipi legittimi,
  KBLOCKS completa il dimensionamento, SUMMARY deve essere
  `structural_checked` con `tensor_count=3` per multi.
- Suite mirata `test_multitensor`: **11 test sintetici** passati in CI,
  incluso big endian, allineamento personalizzato, ordine descrittori,
  offset/sovrapposizione, extent insufficiente, tipo sconosciuto.
  Suite legacy fixture adattate da 3 a 4 modelli, senza modifica ai parser.
- Il flusso Ruth resta a **13 fasi**. Stato MIG-016:
  implementation **complete**, verification **passed (CI)**,
  integration **merged**, personal `GGUF-MULTITENSOR-MANUAL-001`
  **pending**. Per collaudo personale: default Ruth e suite
  `python -m unittest models.formats.gguf.tests.test_multitensor -v`
  (11 test). Se positivo su commit main verificato, MIG-FLOW-002-R2
  autorizza la chiusura automatica senza richiesta di ulteriore consenso.
- Nessun GGUF reale/pesi, tensor_data, hash, dequantizzazione, rete,
  download, CUDA o inferenza. DATA-001/REMOTE-001 invariati, Kaggle inattivo.
  `local_sync: pending_no_access`; nessun perimetro MIG-017 approvato.

## MIG-016 chiusa con collaudo delegato (17 ottobre 2026)

- Evidenza portabile: MIG-016-delegated-001; mode: delegated; status: passed.
  Commit main eseguito: dbdef7ab000d99d2e3dde68b0857840f0047d5fe, risolto via
  heads/main e verificato prima del collaudo. Nessuna esecuzione personale
  dell'utente viene attribuita a questa prova.
- Destinazione: nuova copia locale dedicata designata dall'utente, creata
  esclusivamente dal repository canonico; integrita' degli oggetti Git,
  branch main, revisione e pulizia verificati. Prova minima di scrittura,
  lettura e rimozione superata; ambiente venv dedicato e isolato disponibile
  senza installazioni di sistema o pacchetti aggiuntivi.
- Default Ruth: 13 fasi completate nell'ordine prescritto e riepilogo
  finale positivo, exit code 0. FIXTURE: passed su quattro file sintetici,
  SUMMARY=structural_checked. Catalogo opzionale e SUMMARY successiva
  not_configured: skip previsto, non verifica di modelli reali.
- Suite prescritta models.formats.gguf.tests.test_multitensor:
  11 test, OK, nessun fallimento. La suite verifica anche i tre descrittori
  del modello sintetico multi e i casi negativi previsti dallo scope.
- MIG-FLOW-002 revisione 3: queste prove complete soddisfano il gate
  GGUF-MULTITENSOR-MANUAL-001, mantenendone l'ID storico ma registrando
  mode: delegated. Nessuna valutazione umana aggiuntiva era prescritta;
  nessuna ripetizione personale o seconda conferma necessaria.
- Stato MIG-016: implementation complete; verification passed (CI e
  collaudo locale sintetico delegato); integration merged (PR #96-#99);
  local validation passed; closure complete.
- validation_checkout: verified_on_tested_revision. Dopo la pubblicazione
  di questi soli documenti, pending_docs_fast_forward fino alla sync
  conservativa della nuova destinazione; il collaudo dello SHA sopra
  resta valido solo se il diff successivo e' esclusivamente documentale.
- I checkout precedenti restano separati e invariati: copia pulita
  precedente pending_permissions; copia storica modificata preservata
  ed esclusa dalla sync. Non sono dichiarati sincronizzati.
- DATA-001: output grezzi, cache e altre evidenze reali conservati solo
  localmente in spazi autorizzati ed esclusi da Git. Pubblicati soltanto
  SHA, ID, stati e conteggi. Nessun GGUF reale, peso o tensor_data,
  hash payload, compatibilita' runtime, CUDA, download o inferenza verificati.
- MIG-017 non approvata. Dopo questa chiusura presentare il prossimo
  perimetro funzionale e attendere un nuovo OK prima di implementarlo.
  Le note precedenti di collaudo pending sono cronologia storica.
