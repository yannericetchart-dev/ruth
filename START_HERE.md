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
normale. Per la prima integrazione automatica: PY-001 poi DEP-001, con il
solo controllo minimo di `pip`; l'inventario completo dei requisiti non e'
ancora definito. Il codice e i nuovi test sono candidati finche' il checkpoint
non registra verifica, merge e collaudo dell'orchestrazione.
