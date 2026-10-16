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
