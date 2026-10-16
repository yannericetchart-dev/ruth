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
normale. Flusso integrato corrente: PY-001, DEP-001, DIAG-001 e HW-001 (CPU/RAM), con il
solo controllo minimo di `pip`; l'inventario completo dei requisiti non e'
ancora definito. DIAG usa report preesistenti, senza modifiche host. MAIN-FLOW-001 e' integrata in main tramite PR #17 e verificata da CI.
Il collaudo personale dell'avvio sequenziale senza argomenti e' stato
superato sul commit `3f08cbad9ec06f814031bd1cbcbdc25553bd82ae`
(MAIN-MANUAL-001 complete). Consultare CHECKPOINT.md per il prossimo
perimetro, che richiede una nuova approvazione funzionale.

## Stato DIAG-001

DIAG-001 e' integrata tecnicamente in main tramite PR #22 e verificata
su GitHub. Il collaudo locale DIAG-MANUAL-001 e' stato confermato positivo sulla
revisione `b379d2d8a228f55101eb98a19bfb14e8fc7a4bcb`, con ordine PY -> DEP -> DIAG.
Consultare CHECKPOINT.md per il prossimo perimetro da approvare.

## HW-SCOPE-001 approvato

Perimetro iniziale hardware CPU/RAM read-only autorizzato il
16 ottobre 2026; lo stato verificato, le promozioni e il successivo
collaudo personale HW-MANUAL-001 sono nel CHECKPOINT.md.

## HW-001 integrata tecnicamente (16 ottobre 2026)

Inventario CPU/RAM tramite `hardware.detect()` e quarto passaggio del
flusso `main.py`, dopo PY->DEP->DIAG. Verifiche sintetiche CI superate,
PR #27 integrata in main. HW-MANUAL-001 superato personalmente
sulla revisione `59b26fd6b427e6756ce0285fee60590cb077ebf1` il 16 ottobre 2026.
Prossima migrazione soltanto dopo nuovo perimetro approvato;
consultare CHECKPOINT.md.

## GPU-SCOPE-001 approvato

Il 16 ottobre 2026 e' stato approvato solo il provider NVIDIA con
interrogazione read-only a `nvidia-smi`. AMD/Intel non autorizzati;
lo stato di verifica, integrazione e collaudo e' in CHECKPOINT.md.

## GPU-001 NVIDIA integrata tecnicamente (16 ottobre 2026)

Il quinto passaggio di `main.py` interroga soltanto NVIDIA in sola
lettura; verifica CI superata e PR #33 integrata in main.
GPU-MANUAL-001 e' stata superata personalmente il 16 ottobre 2026
sulla revisione `c344c71013cfa05da4f49f3204900f4bdc89b8ea`; tutte le cinque fasi completate.
La successiva migrazione richiede nuova approvazione del perimetro
(MIG-FLOW-001). AMD/Intel e CUDA non sono migrati.

## CAT-SCOPE-001 approvato

Avvio della porzione Models catalogo locale, approvata il
16 ottobre 2026: manifest JSON opzionale ignorato da Git,
sesto passaggio Ruth; nessuna scansione/download/GGUF/inferenza.
Stato implementazione, CI e collaudo CAT-MANUAL-001 in CHECKPOINT.md.

## CAT-001 integrata tecnicamente (16 ottobre 2026)

`main.py` include come sesto passaggio il catalogo modelli locale
`.local/models/catalog.json` opzionale; manifest assente o vuoto
sono stati validi e non causano fallimento. PR #38 in main e CI
superate; **CAT-MANUAL-001 e' stata superata personalmente**
il 16 ottobre 2026 sul commit `90e57f106fb664b0257775260044125ea85228e2`.
Il prossimo dominio richiede un nuovo perimetro funzionale esplicito,
ancora non approvato (MIG-FLOW-001).

## GGUF-SCOPE-001 approvato (16 ottobre 2026)

L'utente ha approvato GGUF-001, sola lettura dell'intestazione dei file
GGUF dichiarati in CAT, come settima fase Ruth. Le PR #41-#44 hanno
integrato il codice in main, commit `15b106082c4dee37622c61a8afefd248a57464a0`, dopo CI positiva.
**GGUF-MANUAL-001 superato personalmente il 16 ottobre 2026**
sul commit `ab81696dc93bc5d71a208cbcdc9a6f714b462fa9`.
Con catalogo assente il settimo controllo e' stato correttamente
`not_configured`; non e' stata eseguita una lettura di GGUF reale.
Nuova migrazione solo dopo perimetro esplicito approvato.
Nessun download, metadato esteso, peso tensor o inferenza autorizzato.

## GGUF-META-SCOPE-001 approvato (16 ottobre 2026)

L'utente ha approvato una lettura read-only, bounded, dei metadati
tecnici GGUF dichiarati in CAT, come ottava fase dopo GGUF.
Il lavoro e' candidato e non ancora integrato: consultare CHECKPOINT.md.
Nessun download, lettura pesi o inferenza autorizzato.
