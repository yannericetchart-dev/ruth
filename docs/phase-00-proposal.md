# Fase 00 — Specifica preliminare di comportamento automatico

> **Stato:** proposta di progettazione, **non approvata**.
> **Data:** 14 ottobre 2026.
> **Roadmap:** [ROADMAP.md](../ROADMAP.md) (approvata con riserva di modifiche future).
> **Regola:** questo documento propone opzioni e contratti; **non autorizza implementazione**, installazioni, download o benchmark.

## Obiettivo della fase 00

Definire con precisione cosa significa «completamente automatico» per il motore di profiling e quali condizioni devono essere rispettate prima che il risultato possa essere utilizzato da un altro branch.

**Definizione proposta:** dopo la scelta iniziale di una *policy*, Ruth rileva l'ambiente, prepara un piano, sceglie e scarica modelli autorizzati, prova più candidati, calibra una configurazione, la verifica con inferenza reale e pubblica un profilo utilizzabile. Durante l'esecuzione non chiede conferme impreviste: se un'operazione esce dai limiti autorizzati, registra il motivo e si arresta senza dichiarare successo.

Il progetto procede **in ordine, per fasi sottoposte ad approvazione separata**. L'automazione riguarda il software finale, non le nostre scelte durante lo sviluppo.

## Decisioni da prendere adesso (opzioni, non default impliciti)

### D1 — Hardware supportato nella prima versione

- **A: Linux + NVIDIA CUDA**, con errori espliciti senza GPU funzionante.
- **B: Linux + NVIDIA CUDA e CPU**, utilizzando CPU solo se consentita dalla policy.
- **C: Linux + più vendor GPU**, richiede adattatori e test hardware ulteriori.

**Proposta:** B, per non legare il progetto a una GPU sola; preferire CUDA se davvero funzionante, **mai** passare silenziosamente a CPU.

### D2 — Motore iniziale

- **A: un backend iniziale verificato (llama.cpp)**, architettura adapter estendibile.
- **B: più backend implementati fin dall'inizio**, con notevole costo di compatibilità e verifiche.

**Proposta:** A; il nome di un backend non implementato non potrà essere selezionato da `auto`.

### D3 — Politica di selezione del modello

- **A: priorità alla qualità entro vincoli di memoria/disco/latency misurati**; velocità come soglia, non obiettivo assoluto.
- **B: priorità alla velocità reale entro una soglia di qualità**.
- **C: punteggio pesato configurabile**, con soglie e pesi esplicitamente approvati.

**Proposta:** A, ma il test di qualità minimo e le soglie vanno definiti nelle fasi 05/08; «modello più grande» non è sinonimo automaticamente di «migliore».

### D4 — Autonomia dei download

- **A: autorizzazione iniziale** che specifica fonti, budget e accettazione licenze; Ruth può scaricare autonomamente entro quei limiti.
- **B: conferma interattiva per ogni download**, più controllo ma automazione incompleta.

**Proposta:** A, con comando `plan` che mostra il costo previsto. Credenziali personali gestite fuori dal repository; i modelli gated non si aggirano.

### D5 — Comportamento in caso di errori

- **A: fallback automatico soltanto entro la policy** (altro candidato, backend o CPU), registrando ogni passaggio, senza nascondere il fallimento.
- **B: stop immediato al primo errore**.

**Proposta:** A per fallimenti recuperabili; fermarsi sui vincoli hard, su errori di sicurezza e su assenza di candidati validi. Niente runtime pubblicato se il test finale non passa.

### D6 — Budget e uso delle risorse

Policy esplicita prima dell'esecuzione: spazio totale scaricabile, spazio libero minimo, massimo numero di candidati, limite di durata/prove, uso delle GPU, riserva di RAM/VRAM, qualità e latenza minime, permesso rete. **Nessun valore numerico è approvato qui**: discuteremo quelli misurabili quando avremo rilevamento hardware e benchmark disponibili. Nel frattempo `plan` deve poter indicare «dato sconosciuto» senza inventarlo.

## Flusso end-to-end proposto

```text
policy approvata
    ↓
doctor: hardware + ambiente + backend realmente disponibili
    ↓
plan: fonti e shortlist compatibili, costi, permessi e vincoli
    ↓
run: download/cache → verifica integrità → baseline nativa
    ↓
benchmark comparabile → massimo contesto operativo
    ↓
KV / parametri / mapping (solo test pertinenti e compatibili)
    ↓
scelta configurazione → validazione isolata indipendente
    ↓
pubblicazione atomica di runtime_profile
    ↓
altro processo: verifica profilo → load → generate / stream → close
```

Tutti i passaggi registrano risultati e motivazioni. Un arresto o crash non cancella i risultati delle fasi completate. `resume` riutilizza soltanto checkpoint ancora compatibili con identità di macchina, modello, backend e workload.

## Interfacce CLI proposte (nomi da approvare)

| Comando | Effetti consentiti |
| --- | --- |
| `doctor` | Inventario di sola lettura; nessun download e nessuna prova pesante |
| `plan` | Valutazione preliminare di candidati, vincoli e costi; nessun download dei pesi |
| `run` | Esecuzione automatica nei limiti di una policy autorizzata e registrata |
| `resume` | Ripresa di prove interrotte con identità ancora compatibile |
| `verify` | Verifica del runtime pubblicato e smoke test reale, nei limiti della policy |

Le opzioni testuali e il formato della configurazione (TOML/JSON) saranno definiti prima dell'implementazione.

## Contratto minimo del risultato (campi concettuali)

Un profilo operativo pubblico deve includere almeno:

- **schema** e versione del contratto;
- **modello**: repository, revisione immutabile, architettura, quantizzazione, file/shard, hash ove disponibile;
- **runtime**: backend, versione, path dell'adapter, opzioni operative applicate e limiti;
- **hardware**: tipo dispositivo, identificatore risolubile e informazioni di compatibilità;
- **risorse**: contesto operativo validato, RAM/VRAM e workload di verifica;
- **evidenza**: risultati del test nativo e del test indipendente, date, stato e motivo della selezione;
- **integrità**: informazioni sufficienti a rilevare un profilo obsoleto;
- **stato**: nessuna promozione senza `verified` effettivo.

**Separazione:** report sperimentali e checkpoint sono privati della pipeline; l'altro branch legge solo il manifest e importa l'API del runtime. In nessun caso deve importare moduli di profiling, chat, voce o visione per generare testo.

## Stati e invarianti

Stati delle singole prove: `pending`, `running`, `verified`, `failed`, `inconclusive`, `skipped`, `unsupported`, `interrupted`.

- La presenza di una GPU **non** certifica CUDA nel backend.
- Il caricamento di pesi **non** certifica una generazione riuscita.
- La dimensione di un GGUF **non** certifica che entri in VRAM.
- Un OOM va identificato con evidenza, non dedotto genericamente da un crash.
- La migliore configurazione di benchmark **non** è automaticamente stabile in produzione.
- Un fallimento successivo **non** deve sovrascrivere l'ultimo profilo verificato.
- Un'incompatibilità rilevata al caricamento dell'API del runtime produce errore esplicito e non un fallback implicito.

## Requisiti di verifica della fase 00 (prima di qualsiasi codice)

- [ ] L'utente approva esplicitamente la definizione di automazione.
- [ ] L'utente decide D1–D5, modificando liberamente le proposte.
- [ ] L'utente accetta che i valori quantitativi di D6 vengano definiti con i dati delle fasi successive, **sempre prima di eseguire prove**.
- [ ] Sono approvate le responsabilità di `doctor`, `plan`, `run`, `resume`, `verify` e del runtime.
- [ ] È accettata la separazione fra dati sperimentali e profilo operativo verificato.
- [ ] La roadmap resta modificabile; variazioni con impatto sulle scelte già implementate sono documentate e verificate.

**Stato finale della fase 00:** non raggiunto; nessuna opzione D1–D6 è stata ancora approvata.
