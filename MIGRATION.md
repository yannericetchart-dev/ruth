# Ruth: registro della migrazione

Lo scaffold non incorpora il codice applicativo storico. Ogni porzione deve
avere una fonte immutabile, una destinazione, test e una revisione separata.
I vecchi branch vengono conservati. Le misure e gli artefatti specifici di
macchina non sono candidati alla migrazione secondo DATA-001.

## Prima porzione prevista: MIG-001 / PY-001

Revisione funzionale: [PY-SCOPE-001](environment/python/MIGRATION_SCOPE.md),
approvata dall'utente il 16 ottobre 2026. Completare la
porzione fino a main senza ulteriori conferme intermedie (MIG-FLOW-001).

- Fonte: `feat/gemma3-phase28-context-dispatch`, commit
  `4ceea508025650ed64b33190834d0b5c85f69db2`,
  `ruth_runtime/venv_packages.py`, logica di verifica del virtualenv.
- Destinazione: `environment/python/runtime`; contratto pubblico del padre
  in `environment/runtime`; accesso comune dalla radice `runtime`.
- Ambito: ispezione in sola lettura dell'interprete e dell'isolamento della
  `.venv` del progetto. Nessuna installazione, download o importazione del
  lock CUDA specifico. Le versioni dei pacchetti sono una porzione successiva.
- Stato: prima implementazione revisionata (PY-REV-001) e integrata tramite
  [PR #2](https://github.com/yannericetchart-dev/ruth/pull/2), branch
  `environment/python/main`, con stato aggiornato al commit
  `b80520ad903ff18f93398725ea95cd91f51da5d8`. Non promossa ai padri superiori;
  il codice di questo branch main resta lo scaffold.
- Esito della revisione: corretto il caso delle dichiarazioni malformate
  ripetute e verificata la catena reale delle API. Prima promozione
  PY-PROMOTE-001 completata, merge `8c76080df5ba34e354f484c92adf7e132b594c2d`.
  L'albero del merge coincide con la revisione verificata. Da promuovere a
  environment/main e in seguito main.
- Verifica prevista: fixture sintetiche per virtualenv attivo, errato,
  globale, con site-packages condivisi e configurazione illeggibile.

## Porzioni successive

1. Dipendenze generiche e diagnostica dell'ambiente, con requisiti verificati.
2. Inventario hardware, separato dall'abilitazione del backend.
3. Catalogo e lettura GGUF, poi acquisizione Hugging Face.
4. Backend CUDA e llama.cpp/llama-cpp-python dietro la sessione comune.
5. Profilazione con schemi portabili e risultati esclusivamente locali.
6. Conversazione e terminale dal branch storico minimal-chat.

La provenienza puntuale e i conflitti fra implementazioni si risolvono prima
di ogni porzione; non dedurre il comportamento dai nomi dei branch. Le API
proposte possono cambiare con decisione registrata e aggiornamento dei padri.

## Stato PY-001 (16 ottobre 2026)

- Perimetro PY-SCOPE-001 approvato; ispezione Python in sola lettura e ingresso `python3 main.py`.
- Promozione foglia: PR #2 in environment/python/main, commit b80520ad903ff18f93398725ea95cd91f51da5d8.
- Promozione intermedia: PR #3, merge environment/main fd61723874052053f38446ee82eca954870804e1.
- Integrazione principale: PR #8, merge main ce1b0bb47b4ae4ad0da40f3f83c964b66015680f.
- Test su GitHub superati: run 38885972247 e 38886317198 (Python sintetico, API, ingresso ufficiale, mappe). Nessun collaudo macchina nativa.
- `local_sync: blocked_divergence` per il checkout principale osservato su branch di lavoro con modifiche; non sovrascrivere nulla. Il collaudo personale PY-MANUAL-001 e' in attesa della sincronizzazione sicura e della conferma positiva.

## PY-MANUAL-001: esecuzione locale comunicata (16 ottobre 2026)

L'utente ha riportato un'esecuzione positiva dall'ingresso `python3 main.py` con
virtualenv isolato. Non sono pubblicati dati di macchina o log (DATA-001).
Il commit HEAD del checkout provato non e' ancora stato comunicato; la
registrazione dell'accettazione sulla revisione rimane in verifica.
Il vecchio checkout locale con modifiche e' conservato senza interventi.

## Chiusura PY-MANUAL-001 (16 ottobre 2026)

- Accettazione positiva **confermata** dall'utente dopo esecuzione reale
  dell'ingresso ufficiale `python3 main.py`.
- SHA del checkout provato: `ef4654af8ab106ef194af254d9f3884e5cabb6b0`, verificato mediante
  `git rev-parse HEAD`. Il commit include la migrazione gia' integrata
  in main (PR #8).
- Stato conclusivo: PY-001 complete, verification passed, integration merged,
  manual acceptance passed. Gli appunti precedenti che parlavano di SHA
  non noto o collaudo pendente sono storici e ora sostituiti da questa voce.
- La prossima porzione richiede descrizione funzionale e nuovo semaforo verde.
  Nessun dato macchina o log incluso (DATA-001).

## Seconda porzione DEP-001 (16 ottobre 2026)

Perimetro [DEP-SCOPE-001](environment/dependencies/MIGRATION_SCOPE.md)
approvato dall'utente. Pianificazione corrente: dipendenze Python generiche
prima dell'hardware, diagnostica DIAG-001 distinta. Provenienza parziale:
`ruth_runtime/venv_packages.py`, commit storico
`4ceea508025650ed64b33190834d0b5c85f69db2`.
Stato di candidatura: implementato sul ramo work, test e promozioni pendenti.
Non importare lock CUDA o dati host, non installare pacchetti (DATA-001).

## DEP-001: promozioni figlie eseguite

- PR #12 in environment/dependencies/main, merge 4af2050e5fc59477f9ab854173bf915b7933d186.
- PR #13 in environment/main, merge c89c02ac0afa828b2c53ebd922a04331069ba527.
- CI sintetica success su run 38895348468 e 38895479549.
- Integrazione root main e conferma personale DEP-MANUAL-001 da completare.

## DEP-001 integrata in main (16 ottobre 2026)

- DEP-SCOPE-001 autorizzato; sola lettura delle dipendenze Python generiche.
- PR #12, #13 e #15: promozioni foglia, environment/main e main concluse.
  Merge root `a542f9ee9f6d09a0496bf756615d2295a7f59ce0`.
- Test remoti success: 38895348468, 38895479549, 38895836703,
  38895924627; fixture sintetiche e mappe coerenti.
- Stato conclusivo tecnico: implementation complete, verification passed,
  integration merged. Stato accettazione manuale DEP-MANUAL-001: pending.
- Le annotazioni precedenti che parlano di promozioni pendenti sono
  storiche e non descrivono lo stato corrente.
- Non avviare diagnostica avanzata, hardware o altro componente finche'
  DEP-MANUAL-001 non sia confermata; dati host esclusi (DATA-001).

## DEP-MANUAL-001 accettata e MAIN-FLOW-001 approvata (16 ottobre 2026)

- Collaudo DEP-001 positivo sul commit `e197cc86829acf52e326a712331cc7ff6cfef6b5` con
  `.venv/bin/python main.py dependencies pip` (stato present_unverified).
- Nuova decisione MAIN-FLOW-001: `main.py` senza argomenti deve eseguire in
  ordine tutte le porzioni gia' integrate, con blocco dei passaggi dipendenti
  quando il prerequisito fallisce. Sottocomandi dedicati restano per diagnosi.
- Primo flusso candidato: Python isolato poi controllo minimo pip;
  l'elenco completo delle dipendenze non e' ancora definito.
- Nuovo codice in verifica; collaudo manuale dell'avvio default dopo merge,
  prima di una nuova migrazione. Nessun dato host pubblicato.

## MAIN-FLOW-001 integrata tecnicamente (16 ottobre 2026)

- `main.py` senza argomenti esegue ora Python poi dipendenze (solo controllo
  minimo `pip`), con blocco motivato dei passaggi dipendenti se Python
  non conforme e riepilogo; sottocomandi specifici ancora disponibili.
- PR #17 merge `e605a02bdd72441fd4e79d36a943109ee3ba5c57`; GitHub CI `38900998138` superata.
- DEP-MANUAL-001 gia' complete sul commit `e197cc86829acf52e326a712331cc7ff6cfef6b5`.
- Nuova verifica personale dell'avvio *sequenziale* MAIN-MANUAL-001
  ancora pendente, da effettuare dopo sincronizzazione sicura di main.
- Non migrare ulteriori funzioni finche' non arriva conferma positiva.

## MAIN-MANUAL-001 completata (16 ottobre 2026)

- L'utente ha sincronizzato la copia pulita sul commit
  `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae` in fast-forward,
  verificando personalmente HEAD ed eseguendo `.venv/bin/python main.py`
  senza argomenti.
- Esito positivo: controllo Python conforme, successivo controllo minimo
  `pip` presente (stato `present_unverified`), sequenza completata.
- Stato MAIN-FLOW-001: implementato, verificato, integrato e accettato.
  Le note precedenti di collaudo pendente sono storiche.
- Ogni nuovo componente dovra' entrare nella sequenza automatica
  di Ruth e richiedera' l'approvazione del proprio perimetro,
  poi il collaudo personale (MIG-FLOW-001/002).
- Nessun dato reale della macchina o log e' stato versionato (DATA-001).

## DIAG-001: approvata diagnostica read-only (16 ottobre 2026)

- Perimetro [DIAG-SCOPE-001](environment/diagnostics/MIGRATION_SCOPE.md)
  autorizzato esplicitamente; interpretare report PY-001 e DEP-001 senza
  nuove letture host o installazioni.
- Componente `environment/diagnostics`, ingresso principale
  `main.py` in sequenza PY -> DEP -> DIAG. La diagnostica puo'
  spiegare Python anche con DEP saltata.
- Stato iniziale: candidato remoto, test e integrazione main pendenti;
  dopo merge servira' DIAG-MANUAL-001 (MIG-FLOW-002).
- Inventario hardware e diagnostica CUDA nativa restano esclusi.

## DIAG-001 integrata in main (16 ottobre 2026)

- Perimetro DIAG-SCOPE-001 esplicitamente approvato dall'utente.
- Contratto puro di diagnostica sui report PY-001 e DEP-001, sequenza
  `main.py` ora PY -> DEP -> DIAG. Se PY non e' isolato, DEP si ferma
  ma DIAG spiega i motivi in base al report esistente.
- Promozioni: PR #20 (`ec7eba74bc71e62ea8562fa51d62295f5f3c01b8`),
  PR #21 (`69ae0821cc0043bd76254bc46dff12b40d0d7b30`),
  PR #22 (`dfdb41e981f32ee233320f77998aadab55ce4d7b`).
- Verifica CI run 38908192677, 38908269925 e 38908370200: success.
- Stato tecnico: implemented / verified / integrated; collaudo personale
  DIAG-MANUAL-001 pending fino a test utente sulla revisione main.
- Niente CUDA, driver, log nativi, installazioni o dati reali in Git.

## DIAG-MANUAL-001: chiusura confermata (16 ottobre 2026)

- Collaudo positivo comunicato dall'utente per `main.py` avviato
  senza argomenti sulla revisione `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`, verificata
  con `git rev-parse HEAD` dopo fast-forward della copia pulita.
- Ordine PY-001 -> DEP-001 -> DIAG-001 rispettato; diagnostica
  `ok_unverified` e riepilogo finale positivo.
- Stato DIAG-001: implementation complete, verification passed,
  integration merged, manual acceptance passed. Le annotazioni di
  attesa precedenti sono storiche e non rappresentano lo stato corrente.
- Nessun dato specifico della macchina viene pubblicato (DATA-001).
  Prossima migrazione da approvare separatamente (MIG-FLOW-001).

## HW-SCOPE-001: prima porzione hardware autorizzata (16 ottobre 2026)

L'utente ha approvato CPU e RAM e i contratti HardwareReport/ResourceReport,
riservando NVIDIA, AMD e Intel GPU a migrazioni successive. Sorgente:
nuova implementazione stdlib portabile; nessun dato host storico migrato.
Destinazione `hardware/cpu` e `hardware/runtime`, integrazione sequenziale
in `main.py`, test sintetici e promozioni foglia fino a main.
Perimetro [hardware/MIGRATION_SCOPE.md](hardware/MIGRATION_SCOPE.md).
Stato candidatura: implementato, non ancora testato o integrato.

## HW-001: implementazione hardware integrata (16 ottobre 2026)

- Perimetro HW-SCOPE-001 approvato: CPU logiche/architettura e RAM
  fisica totale (byte) quando osservabile; GPU esclusa.
- PR #25 in hardware/cpu/main, merge `bf8c72e4566c22b768b36b770c66bbd0f770dbaa`;
  PR #26 in hardware/main, merge `d2ea16b14b89cefacea574f1e209fecec29f7f01`;
  PR #27 in main, merge `35a6ac79a1b1287090c1a3d1dc9208d4129a15ec`.
- CI GitHub 38913317440, 38913438829, 38913533057: success,
  74 test sintetici e 249 mappe. Punto di ingresso Ruth:
  PY -> DEP -> DIAG -> HW. Osservazioni hardware non salvate in Git.
- Implementazione e verifica automatica completate; collaudo personale
  HW-MANUAL-001 **pending** sulla revisione main dopo sync sicura.
- Annotazioni storiche di candidato non integrato restano storiche.

## HW-MANUAL-001 concluso (16 ottobre 2026)

- L'utente ha confermato l'aggiornamento fast-forward della copia
  pulita di main fino al commit `59b26fd6b427e6756ce0285fee60590cb077ebf1`, verificato
  mediante `git rev-parse HEAD`.
- `.venv/bin/python main.py` ha eseguito positivamente
  PY -> DEP -> DIAG -> HW; rilevamento CPU e RAM stato available
  e riepilogo terminale conclusivo positivo.
- Stato HW-001/CPU-001: implementation complete, verification passed,
  integration merged, manual acceptance passed. Le note precedenti
  sul collaudo in attesa sono storiche, non lo stato corrente.
- Nessun valore o log reale di host in Git (DATA-001).
  I provider GPU e altri componenti richiedono nuovo perimetro
  funzionale approvato prima dell'implementazione.

## GPU-SCOPE-001: prima porzione GPU NVIDIA (16 ottobre 2026)

Perimetro [GPU-SCOPE-001](hardware/gpu/MIGRATION_SCOPE.md) approvato.
Branch foglia `hardware/gpu/nvidia/work/GPU-001`; provider nvidia-smi
read-only con modello e VRAM, senza driver/CUDA/inferenza, AMD/Intel non
migrati. Punto di ingresso `main.py` con sequenza PY -> DEP -> DIAG ->
HW -> NVIDIA. CI e promozioni ancora pendenti. Accettazione personale
GPU-MANUAL-001 richiesta solo dopo main, secondo MIG-FLOW-002.

## GPU-001 NVIDIA integrata (16 ottobre 2026)

- GPU-SCOPE-001 esplicitamente approvato: solo NVIDIA e read-only.
- PR #30 `hardware/gpu/nvidia/main` merge
  `21c17032af358120b7cf0d56246a9c135f2833f7`;
  PR #31 `hardware/gpu/main` merge
  `0010c43b75c1f28c18b6c83dba0c913f384a40c2`;
  PR #32 `hardware/main` merge `a7c5e63ea460c00d9ff5c73913af3835c182b2e2`;
  PR #33 `main` merge `54a02cf37262a78f5a1ed0aa84cd6d210060a14c`.
- CI GitHub 38918374871, 38918650589, 38918759116 e 38918867451:
  success, 92 test sintetici e 258 mappe.
- Default `main.py` e' ora PY->DEP->DIAG->HW CPU/RAM->NVIDIA,
  con errori NVIDIA non bloccanti e nessun controllo driver/CUDA.
- Stato tecnico complete/passed/merged; collaudo personale
  GPU-MANUAL-001 **pending** (MIG-FLOW-002). Annotazioni precedenti
  della candidatura non ancora integrata sono storiche.

## GPU-MANUAL-001 completata (16 ottobre 2026)

- L'utente ha sincronizzato in fast-forward il clone pulito main
  alla revisione `c344c71013cfa05da4f49f3204900f4bdc89b8ea`, confermata da `git rev-parse HEAD`.
- Esecuzione personale `.venv/bin/python main.py` riuscita con
  tutti e cinque i passaggi in ordine PY -> DEP -> DIAG -> HW -> NVIDIA.
  Provider NVIDIA con stato `available`, riepilogo positivo.
- Stato GPU-001: implementation complete, verification passed,
  integration merged e manual acceptance passed. Le note precedenti
  sulla verifica manuale pendente restano storiche.
- Non sono versionati inventari, modelli, misure di memoria o log
  host (DATA-001). Prossima migrazione solo previo scope approvato.

## CAT-SCOPE-001: catalogo locale autorizzato (16 ottobre 2026)

- Scopo approvato [CAT-SCOPE-001](models/catalog/MIGRATION_SCOPE.md):
  unicamente catalogo JSON opzionale locale e lettura di riferimenti
  dichiarati in area `.local/models` (senza download/scansioni).
- API e contratti nuovi read-only, in `models/catalog/runtime`; nessun
  codice di acquisizione/GGUF/inferenza migrato implicitamente.
- `main.py` candidato PY->DEP->DIAG->HW->NVIDIA->CAT.
- Stato candidato: implementato su ramo lavoro, CI e promozioni
  pendenti; CAT-MANUAL-001 dopo merge (MIG-FLOW-002).

## CAT-001 integrata in main (16 ottobre 2026)

- Scope CAT-SCOPE-001 approvato dall'utente e immutato: manifest
  JSON locale opzionale, sola lettura, nessuno scanning/download/GGUF.
- PR #36 `models/catalog/main` merge
  `a562e1b4ce948d3865a15a9d06454d91b8cb7cde`;
  PR #37 `models/main` merge
  `77e3a7327c9f0750d6e8363766000ba830ea1c5e`;
  PR #38 main merge `12c395ba3bb272d66095d4e468794936b1693892`.
- CI 38928909391, 38929028829, 38929118836: success
  con 116 test sintetici e 264 mappe; default Ruth continua
  i cinque passaggi precedenti e aggiunge CAT per sesto.
- Stato tecnico implemented/verified/integrated; collaudo personale
  CAT-MANUAL-001 pendente, quindi prossimo componente bloccato.
- Nessuna informazione reale su host/modelli/versioni in Git (DATA-001).

## CAT-MANUAL-001: chiusura personale (16 ottobre 2026)

- L'utente conferma personalmente esito positivo di `.venv/bin/python main.py`
  senza argomenti sul commit main `90e57f106fb664b0257775260044125ea85228e2`.
- Sequenza PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT completa; catalogo
  opzionale assente con stato `not_configured` conforme al perimetro.
- CAT-001: implementazione completa, CI sintetica superata, PR #38 in main
  e collaudo CAT-MANUAL-001 passato. Le precedenti voci di attesa sono storiche.
- Il prossimo componente rimane da approvare secondo MIG-FLOW-001/002;
  dati locali esclusi dal repository secondo DATA-001.

## GGUF-SCOPE-001 approvato (16 ottobre 2026)

- Scopo circoscritto alla lettura dei 24 byte dell'intestazione di soli GGUF
  esplicitamente dichiarati, con codici portabili; nessuna lettura dei pesi.
- Destinazione `models/formats/gguf/runtime`, settimo controllo di `main.py`.
- Approvazione utente acquisita, lavoro remoto candidato; test e promozioni
  pendenti. Dopo main obbligatorio GGUF-MANUAL-001 personale.
- Perimetro: [GGUF-SCOPE-001](models/formats/gguf/MIGRATION_SCOPE.md).

## GGUF-001: integrazione in main (16 ottobre 2026)

- Autorizzazione GGUF-SCOPE-001 rispettata: solo primi 24 byte dei
  file GGUF espressamente dichiarati da CAT, senza payload o installazioni.
- Promozioni PR #41, #42, #43 e #44 fino a main `15b106082c4dee37622c61a8afefd248a57464a0`;
  CI sintetiche e mappe verificate con successo in tutti i passaggi.
- GGUF-MANUAL-001 non ancora svolto; prossimo passo esecuzione personale
  di `main.py` senza argomenti sul checkout principale sincronizzato.
- Le note di candidato non verificato/non integrato sono storiche e
  sostituite dalla presente registrazione. DATA-001 e REMOTE-001 invariati.

## GGUF-MANUAL-001: chiusura collaudo (16 ottobre 2026)

- L'utente ha verificato HEAD ed eseguito personalmente Ruth senza
  argomenti da main `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`;
  successivamente ha confermato esplicitamente esito positivo.
- PY -> DEP -> DIAG -> HW -> NVIDIA -> CAT -> GGUF completate.
  Catalogo e intestazioni GGUF `not_configured` sono risultati validi.
  Questa prova non comprende un file GGUF reale.
- GGUF-001 completa, CI superata, integrata e accettata; note di
  attesa precedenti mantenute come storia, non come stato corrente.
- Nessun nuovo componente approvato. DATA-001 e REMOTE-001 invariati.

## GGUF-META-SCOPE-001: approvato (16 ottobre 2026)

- Lettura tecnica bounded di coppie GGUF metadata per sole dichiarazioni
  CAT: architecture, file_type numerico e context_length opzionale.
- API per id e catalogo, nuovo ottavo passo di Ruth e comando mirato.
- Nessuna lettura di payload tensor o uso modello; solo fixture
  sintetiche in CI. Candidato remoto in verifica, non ancora main.
- Promozioni gguf -> formats -> models -> main approvate senza
  nuovi semafori intermedi. Collaudo manuale post-merge obbligatorio.
- Scope: [GGUF-META-SCOPE-001](models/formats/gguf/METADATA_SCOPE.md).

## GGUF-META-001: integrazione tecnica completata (16 ottobre 2026)

- GGUF-META-SCOPE-001: lettura limitata dei tre metadati dichiarati
  di soli GGUF del catalogo, senza tensor, download o inferenza.
- PR #47, #48, #49, #50 promosse lungo tutti gli antenati fino
  a main `a630c83f6c62a229e18fe5e64f0863c19de16018`; CI e test sintetici/mappe superati.
- L'ottava fase default di Ruth e il comando mirato sono integrati.
- GGUF-META-MANUAL-001 ancora pendente: necessita esecuzione
  personale positiva e conferma esplicita sul main sincronizzato.
- Gli stati precedenti di candidato sono annotazioni storiche.

## Chiusura GGUF-META-MANUAL-001 (16 ottobre 2026)

- Esito positivo confermato esplicitamente dall'utente dopo avvio
  personale di Ruth dal main `d318fd793da514f21ee2e8475304d1a6c883ee94`.
- Sequenza completa di otto controlli; catalogo e GGUF/META
  `not_configured` sono skip validi, senza download o inferenza.
- GGUF-META-001: implementazione completa, CI sintetica superata,
  PR #50 integrata e collaudo personale passato.
- Le voci anteriori di manual pending restano storiche.
  Prossima migrazione solo previo nuovo scope e approvazione.

## GGUF-INDEX-SCOPE-001: nona porzione autorizzata (16 ottobre 2026)

- Solo descrittori tensor di GGUF catalogati; lettura bounded,
  nome/dimensioni/codice tipo/offset, no payload.
- Nuove API inspect_tensor_index e inspect_all_tensor_indexes;
  nona fase default Ruth e comando mirato `gguf-index`.
- Verifica dimensionale solo per tipi scalari noti; altri tipi
  registrati come size_unverified.
- Scope [GGUF-INDEX-SCOPE-001](models/formats/gguf/INDEX_SCOPE.md).
- Candidato remoto; CI/promozioni pendenti, poi test personale
  GGUF-INDEX-MANUAL-001. Nessuna fase successiva autorizzata.

## GGUF-INDEX-001: integrazione tecnica (16 ottobre 2026)

- Scope bounded read-only di soli descriptor dell'indice tensori GGUF
  dichiarati nel catalogo; nessun tensor payload letto.
- CI sintetica, regressioni e mappe passate ai quattro livelli.
  PR #53, #54, #55, #56 fino a main `1d51cfeed2bec83119ebf56ff0cb3840089484cf`.
- Ruth default comprende ora nove fasi, con INDEX dopo META;
  stato not_configured valido se manca il catalogo.
- GGUF-INDEX-MANUAL-001 pending: utente dovra' eseguire main.py
  di persona da main sincronizzato e confermare esito positivo.
- Le annotazioni candidate precedenti restano storiche; nessuna
  porzione successiva approvata implicitamente.
