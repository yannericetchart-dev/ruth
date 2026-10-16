# Ruth: struttura evolvibile, revisione 1

La struttura e le firme proposte sono una base di lavoro. Una funzione
documentata come proposta non e' una funzione disponibile. I branch storici
restano fonti della migrazione e non vengono riscritti.

## Nodi e corsie

I domini sono environment, hardware, models, inference, profiling e chat.
`runtime/` alla radice ospita contratti e orchestrazione comuni. Ogni dominio
separa implementazioni (`runtime`), verifiche (`tests`), codice di misura
(`benchmarks`) e strumenti (`tools`). Le corsie dei componenti piu profondi
si aggiungono quando servono. Uno strumento GGUF appartiene a
`models/tools/gguf`; il lettore riutilizzabile a `models/formats/gguf`.
La sua implementazione potra stare in `models/tools/gguf/runtime` senza
perdere la classificazione tools ereditata dagli antenati.

Ogni cartella di progetto ha ROADMAP, CODEMAP e APIMAP. Ogni componente
autonomo ha anche NODE.yaml. Le directory nascoste di infrastruttura e quelle
ignorate locali non sono nodi architetturali. Il file main.py e' ora l'ingresso ufficiale minimo per l'ispezione
read-only Python (PY-001); le altre funzioni applicative restano pianificate.

## Fonte dei metadati e aggiornamento dei padri

NODE.yaml e' scritto nel sottoinsieme JSON di YAML, leggibile senza dipendenze.
Contiene responsabilita, priorita, ID stabili, dipendenze, API e stati.
I tre documenti Markdown sono viste generate dei metadati. Non modificarli
isolatamente: modificare il nodo interessato, aggiornare le attivita dipendenti
e le decisioni, poi eseguire dalla radice:

```sh
python3 tools/maps.py --write
python3 tools/maps.py --check
```

La rigenerazione aggiorna tutti gli antenati. Un cambio di dipendenza puo
richiedere anche modifiche semantiche alle attivita del padre: il generatore
non decide la strategia. Se lavorando su chat emerge una necessita in models,
aggiungere li un'attivita con ID stabile, inserire tale ID in blocked_by della
chat e aggiornare il checkpoint prima di proseguire sul lavoro dipendente.
Il validatore rileva genitori incoerenti, cicli tra dipendenze, ID duplicati,
stati invalidi e mappe non aggiornate.

## Stati indipendenti

| Campo | Valori | Significato |
| --- | --- | --- |
| development | inactive, active, paused, maintenance, retired | Lavoro corrente |
| implementation | unassessed, planned, skeleton, partial, complete | Codice nella nuova struttura |
| maturity | null, experimental, candidate, stable | Qualita del contratto/implementazione |
| verification | unknown, not_run, passed, failed, stale | Validita della verifica dichiarata |
| integration | null, not_merged, partial, merged | Integrazione nel padre |
| branch.status | planned, present, archived | Esistenza del branch |
| flags | in_scope, future, experimental, altri tag espliciti | Classificazioni aggiuntive |
| blocked_by | ID delle attivita bloccanti | Blocchi separati dallo sviluppo |

`passed` richiede indicazione dell'ambito verificato nel checkpoint del nodo;
non equivale a verifica nativa o stabilita. I risultati reali rimangono locali.
Un figlio futuro/inattivo non blocca il completamento dell'ambito attivo.
Per attivarlo servono decisione, aggiornamento del perimetro e nuova roadmap.
La qualifica experimental puo descrivere il contratto o un flag di lavoro,
ma non sostituisce il campo development.

## Branch Git

Il padre logico non e' un contenitore Git: ogni branch contiene un intero
snapshot. Usare nomi foglia `environment/main`, `environment/python/main`,
`inference/engines/llama.cpp/main` e
`inference/engines/llama.cpp/bindings/llama_cpp_python/main`.
I branch di lavoro terminano con `/work/<attivita>`. Il NODE specifica il
padre di integrazione. Non creare un ref `inference` insieme a `inference/...`:
Git non consente quel conflitto tra file e directory.

Un ramo planned non viene creato solo perche compare nella mappa. AMD,
Intel GPU, Kaggle e gli altri provider futuri restano inattivi. Si creano
branch di sottosistema quando inizia il lavoro, conservando i branch storici.
Promuovere il codice dal figlio al padre con revisioni esplicite; ogni API
pubblica del figlio deve soddisfare il contratto comune del padre. Il nome
del branch da solo non impone nessuna compatibilita.

## Ordine di migrazione e dipendenze

1. Ambiente Python e contratti minimi necessari (PY-001 integrata e accettata).
2. Dipendenze Python generiche in sola lettura (DEP-001); diagnostica avanzata separata.
3. Inventario hardware e disponibilita delle risorse.
4. Catalogo, formati e acquisizione da Hugging Face.
5. Backend CUDA, motore llama.cpp, binding llama-cpp-python e sessione comune.
6. Profilazione: selezione, memoria, contesto, placement, precisione, profili.
7. Conversazione e interfaccia terminale.

Inference dipende da models e hardware; profiling da models, hardware e
inference; chat da inference. Il download non richiede una GPU. Le dipendenze
native di un motore appartengono ai suoi tools, non all'ambiente generico.
Migrare una porzione per volta con provenienza, test pertinenti, checkpoint
del componente e propagazione fino alla radice. Non copiare risultati,
inventari di macchina, lock specifici dell'host o registrazioni reali.

Prima di ogni porzione sottoporre all'utente la descrizione delle funzioni
e attendere il verde sul perimetro (MIG-FLOW-001). Dopo l'approvazione,
completare tutte le promozioni fino a main senza chiedere conferma a ogni
passaggio; fermarsi per dubbi o domande che richiedono una decisione.

Ogni porzione deve inoltre essere eseguita dal punto d'ingresso ufficiale
di Ruth in main con messaggi a terminale corrispondenti (MIG-FLOW-002).
Dopo l'integrazione, l'utente esegue Ruth localmente di persona e conferma
l'esito positivo prima della migrazione successiva. L'integrazione tecnica
e il collaudo manuale sono due stati distinti; test automatici e silenzio non
sostituiscono la conferma dell'utente.

## Punto d'ingresso completo e sequenziale (MAIN-FLOW-001)

L'esecuzione normale di Ruth usa `main.py` senza argomenti. L'ingresso
ordina e avvia automaticamente tutte le funzioni approvate e integrate,
senza richiedere un sottocomando per ogni porzione. Ogni migrazione futura
deve collegarsi a questo flusso, aggiungendo test di ordinamento e di
comportamento quando un prerequisito non e' valido. I sottocomandi restano
per controlli isolati. Finche non esiste un elenco approvato delle
dipendenze dell'app, l'autoverifica DEP-001 controlla solo `pip`, descritto
come controllo minimo e non come inventario o certificazione di compatibilita'.
