# Ruth: checkpoint di progetto

Revisione documentale: 32 (indice main); CONT-001 revisione 1.
Fase 3: migrazione incrementale per componente.
Stato: **GGUF-INDEX-001 implementata, verificata in CI, integrata in main
e accettata personalmente (GGUF-INDEX-MANUAL-001 complete)**.
GGUF-MANUAL-001 e CAT-MANUAL-001 restano complete.
PY, DEP, DIAG, HW/CPU e GPU NVIDIA restano accettate personalmente.
L'ingresso senza argomenti di Ruth esegue tutte le funzioni integrate.

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
