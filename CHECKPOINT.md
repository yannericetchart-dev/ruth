# Ruth: checkpoint di progetto

Revisione documentale: 25 (indice main); CONT-001 revisione 1.
Fase 3: migrazione incrementale per componente.
Stato: **CAT-001 implementata, verificata e integrata in main;
collaudo personale CAT-MANUAL-001 ancora da confermare**.
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

MIG-006 e' bloccata da CAT-MANUAL-001: il catalogo locale e' tecnicamente
integrato ma non ancora collaudato personalmente. La prossima
migrazione richiede **prima** un perimetro funzionale esplicito
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
