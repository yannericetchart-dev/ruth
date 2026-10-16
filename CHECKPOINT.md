# Ruth: checkpoint di progetto

Revisione documentale: 5 (branch di lavoro).
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

PY-001 e' implementata nel branch `environment/python/work/inspect-venv`.
Ambito: ispezione Python in sola lettura; contratto environment e accesso
comune runtime, con verifiche sintetiche. Vedere environment/CHECKPOINT.md
e environment/python/CHECKPOINT.md. Non integrata nei branch padri.

Revisionare e promuovere la porzione lungo environment/python/main,
environment/main e main, aggiornando i checkpoint ad ogni promozione.
Le altre porzioni rimangono pianificate; Kaggle resta inattivo.

## Limiti correnti

- Nel branch main rimane lo scaffold senza migrazione applicativa; il codice
  di PY-001 e' confinato al branch di lavoro fino alla revisione.
- Non avviare il passaggio 3 prima della verifica dei passaggi precedenti.
- Nessuna misurazione nativa e' certificata da questo checkpoint.
- Branch e commit delle attivita' successive saranno indicati nei checkpoint
  relativi; i riferimenti di revisione sono Git, non copie di dati di host.
