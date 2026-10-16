# Ruth: punto di ingresso comune

Repository canonico: <https://github.com/yannericetchart-dev/ruth>.
Branch di ingresso: `main`. Protocollo: `CONT-001`, revisione 1.

Questo indice e' comune a ChatGPT e Codex. Leggerlo nuovamente da GitHub
all'inizio di ogni attivita', fissando la revisione letta. Le copie locali e
le fonti caricate possono essere precedenti; confrontarle prima di operare.
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
Se manca una fonte o un collaudo non e' stato svolto, dichiararlo. Non
dedurre il completamento di una fase dalla sola presenza dei documenti.
