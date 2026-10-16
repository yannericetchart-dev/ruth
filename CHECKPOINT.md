# Ruth: checkpoint di progetto

Revisione documentale: 10 (indice main; protocollo REMOTE-001).
Attivita': `MIG-001` (prima porzione: `PY-001`).
Fase corrente: **3 - migrazione incrementale per componente**.

## Completato

- Definito il punto di ingresso comune in GitHub e il protocollo CONT-001.
- Registrati il vincolo DATA-001 e l'esclusione di Kaggle dallo sviluppo.
- Preparati istruzioni Codex, istruzioni da applicare al Progetto ChatGPT
  e procedura di verifica.
- Pubblicati i documenti comuni nel branch main.
- Verificato con un nuovo processo diagnostico Codex che le istruzioni di
  progetto entrino nel contesto iniziale, inclusi CONT-001 e DATA-001.
  Questa verifica riguarda il caricamento, non la risposta di un modello.
- Verificata l'esclusione Git dei percorsi convenzionali per artefatti locali.

## Collaudo SETUP-001 completato

- Salvate le istruzioni nel Progetto ChatGPT.
- Una nuova sessione Codex e una nuova chat ChatGPT hanno letto da GitHub
  l'ingresso alla revisione `b59dc7b3477918ffc4a11e520608f301d50ac155`.
  Entrambe hanno ricostruito fase, prossima azione e vincoli CONT-001,
  DATA-001 e Kaggle inattivo. Evidenze: `SETUP-001`, conservate solo localmente.
- Codex ha confrontato il checkout. ChatGPT ha dichiarato l'indisponibilita'
  del checkout nella sessione: l'accesso locale non deriva dalla sincronizzazione
  Git e non e' un requisito per la lettura della fonte GitHub.
- Il passaggio 1 e' completato. Ripetere il collaudo se cambia il protocollo.

## Prossima azione

Il passaggio 2 e' completato: scaffold dei domini, corsie, metadati di stato,
mappe ricorsive e controllo di coerenza. Ambito della verifica: struttura e
propagazione dei riepiloghi, non esecuzione applicativa o nativa.

Prima porzione Python integrata in `environment/python/main` tramite
[PR #2 integrata](https://github.com/yannericetchart-dev/ruth/pull/2), merge
`8c76080df5ba34e354f484c92adf7e132b594c2d`.
Stato aggiornato del componente e relative mappe al commit
`b80520ad903ff18f93398725ea95cd91f51da5d8`.
La revisione di codice esaminata e' `80218739327666b0250c3715ba9cfb6fe1f114d5`.
environment/main e main conservano ancora lo scaffold applicativo.

Revisione tecnica PY-REV-001 completata: corretto un caso in cui una
dichiarazione malformata ripetuta veniva ignorata; verificati gli ingressi
pubblici reali e l'ambito obbligatorio delle verifiche nelle mappe.
Nessun problema bloccante residuo nell'ambito della porzione.

PY-PROMOTE-001 completata: il merge conserva la storia e produce lo stesso
albero della revisione che ha superato i test. Le evidenze restano locali.
Il campo integration del nodo Python resta not_merged rispetto al suo padre
environment/main: la prima promozione al branch canonico e' invece completata.

Passaggio corrente: revisione funzionale del perimetro
[PY-SCOPE-001](environment/python/MIGRATION_SCOPE.md) da parte dell'utente.
Semaforo verde su questa descrizione: **concesso dall'utente il 16 ottobre 2026**. Il codice del punto d'ingresso e' candidato in PR #3; le suite nuove e le promozioni fino a main non sono ancora completate.

Secondo REMOTE-001, completare direttamente su GitHub le promozioni da
environment/python/main a environment/main e quindi main, con test
in snapshot isolati e pubblicazione remota delle correzioni,
revisioni e aggiornamenti delle mappe, senza ulteriori conferme intermedie.
Fermarsi per dubbi o domande che richiedono una decisione dell'utente,
secondo MIG-FLOW-001. Consultare il
[checkpoint immutabile](https://github.com/yannericetchart-dev/ruth/blob/b80520ad903ff18f93398725ea95cd91f51da5d8/environment/python/CHECKPOINT.md)
e aggiornare lo stato dopo ogni promozione.

MIG-FLOW-002 aggiunta: per PY-001 il percorso finale in main deve eseguire
l'ispezione dal punto d'ingresso di Ruth con messaggi a terminale relativi
alla funzione e al suo esito. Il collegamento e' implementato nella PR #3, ma non verificato o integrato in main.
Dopo l'integrazione fornire comando ed esiti attesi all'utente, che avviera
Ruth localmente di persona. PY-MANUAL-001: pianificata, nessuna conferma
positiva registrata. Non iniziare la migrazione successiva finche l'utente
non conferma l'esito positivo del collaudo sulla revisione provata.

Ambito verificato: test sintetici dell'isolamento Python e API comuni;
nessuna certificazione del runtime nativo. I risultati restano locali.
Le altre porzioni rimangono pianificate; Kaggle resta inattivo.


## Protocollo operativo corrente (16 ottobre 2026)

- REMOTE-001 accettata e prevalente su SYNC-001 (storica e superseded):
  ChatGPT e Codex pubblicano il codice versionato direttamente su GitHub;
  il clone principale locale non si modifica durante lo sviluppo.
- Test e generatori locali sono ammessi soltanto su copie di snapshot GitHub
  in aree temporanee esterne o in `.local/verification/` ignorata;
  verificare sempre la revisione remota candidata. Nessuna scrittura
  ai file di lavoro del clone principale durante il ciclo remoto.
- Le modifiche locali preesistenti viste su
  `environment/python/work/entrypoint` sono fuori dal perimetro corrente,
  non vengono importate, pubblicate o cancellate. Il loro stato non blocca
  la prosecuzione remota di PY-001, gia' approvata.
- PR #3: candidato PY-001 per l'ingresso Ruth; test nuovi, mappe e promozioni
  fino a main ancora da completare. Non dichiarare PY-001 integrata.
- `local_sync: pending_final_merge`. Dopo il merge completo verificare
  l'allineamento del checkout reale o fornire istruzioni di sync sicura;
  se il checkout e' sporco o su altro branch, conservarlo e segnalare
  `blocked_divergence`, senza scarti automatici.
- PY-MANUAL-001: collaudo personale dopo la sync post-merge ancora in
  attesa; non iniziare la migrazione seguente senza esito positivo confermato.
- La regola va verificata su una nuova sessione Codex; l'AGENTS.md locale
  potrebbe non contenere ancora questa revisione finche non avviene la sync.

## Limiti correnti

- In main non e' stato integrato codice applicativo. PY-001 e' disponibile
  in environment/python/main e non deve essere descritta come disponibile
  in environment/main o main.
- Non iniziare la migrazione successiva finche PY-001 non e' integrata, sincronizzata localmente e collaudata personalmente secondo MIG-FLOW-002.
- Nessuna misurazione nativa e' certificata da questo checkpoint.
- Branch e commit delle attivita' successive saranno indicati nei checkpoint
  relativi; i riferimenti di revisione sono Git, non copie di dati di host.
