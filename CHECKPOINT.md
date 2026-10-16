# Ruth: checkpoint di progetto

Revisione documentale: 6 (indice main).
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

Prima porzione Python implementata e verificata nel branch
`environment/python/work/inspect-venv`, commit
`80218739327666b0250c3715ba9cfb6fe1f114d5`:
[PR #2 pronta per la revisione finale](https://github.com/yannericetchart-dev/ruth/pull/2).
Destinazione immediata: `environment/python/main`, poi `environment/main`,
infine `main`. Tutti i branch padri conservano ancora lo scaffold.

Revisione tecnica PY-REV-001 completata: corretto un caso in cui una
dichiarazione malformata ripetuta veniva ignorata; verificati gli ingressi
pubblici reali e l'ambito obbligatorio delle verifiche nelle mappe.
Nessun problema bloccante residuo nell'ambito della porzione. La PR non e'
piu in bozza e nessun merge e' stato eseguito.

Prossimo passaggio: prima promozione a environment/python/main. Consultare
il [checkpoint immutabile](https://github.com/yannericetchart-dev/ruth/blob/80218739327666b0250c3715ba9cfb6fe1f114d5/environment/python/CHECKPOINT.md)
prima dell'integrazione; aggiornare lo stato dopo ogni promozione.
Ambito verificato: test sintetici dell'isolamento Python e API comuni;
nessuna certificazione del runtime nativo. I risultati restano locali.
Le altre porzioni rimangono pianificate; Kaggle resta inattivo.

## Limiti correnti

- In main non e' stato integrato codice applicativo. PY-001 e' confinata al
  branch di lavoro e non deve essere descritta come disponibile in main.
- Non avviare il passaggio 3 prima della verifica dei passaggi precedenti.
- Nessuna misurazione nativa e' certificata da questo checkpoint.
- Branch e commit delle attivita' successive saranno indicati nei checkpoint
  relativi; i riferimenti di revisione sono Git, non copie di dati di host.
