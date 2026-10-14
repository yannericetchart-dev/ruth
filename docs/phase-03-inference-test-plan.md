# Ruth — Piano di qualificazione dell'inferenza CUDA (Fase 03)

**Stato:** proposta tecnica, non ancora implementata o approvata come comando CLI.
**Prerequisito verificato sul PC:** 211 test OK e
`./ruth setup --verify-native-library` → `native_library_dual_abi_verified`.
L'utente ha verificato il byte-match del wheel con tutte le librerie .so
installate, C ABI CUDA e stesso `libllama.so` dal binding Python.
**Non** sono stati caricati modelli o prodotti token reali.

## Obiettivo

Un primo test di inferenza reale GGUF, isolato, ripetibile e fail-closed,
usando `NVIDIA + CUDA + llama.cpp + llama_cpp_python` (profilo già stabilito).
Non è una sessione chat e non sostituisce il profiling/benchmark completo.

## Preflight per esecuzione (non scarica pesi)

- Identificare un **GGUF locale** completo e di provenienza verificabile;
  fonte/revisione/versione/quantizzazione/licenza, dimensione e SHA-256
  devono essere tracciati, senza supporli dal solo nome del file.
- Non scaricare nulla senza previa decisione su fonte, licenza e budget.
  Se non c'è un GGUF idoneo, produrre piano e blocker: non falsificare
  una prova con un modello simulato.
- Validare il file e la disponibilità di VRAM, RAM e disco, la GPU PCI
  selezionata, driver/CUDA e lo stato del runtime pinning. Rifiutare
  symlink non previsti, file non GGUF, hash differente, schema inatteso.
- Non rilanciare build o installazione già completate. Nessun wheel di
  fallback o inferenza CPU.

## Worker isolato

Eseguire in un **subprocess della sola .venv** (timeout e codice uscita
espliciti), non importare il motore dentro il processo orchestratore.

- `CudaLlamaAdapter` con `n_gpu_layers=-1` e `Llama.cache is None`;
  disabilitare `LlamaDiskCache`. Usare parametri fissati/registrati:
  `n_ctx`, `n_batch`, `n_ubatch`, temperatura e seed.
- Prima del carico e durante il worker, campionare memoria GPU e
  utilizzo tramite fonte disponibile (`nvidia-smi`) e correlare la GPU
  osservata a quella selezionata e al PID del worker quando possibile.
  Valori pre/post da soli non dimostrano un offload: serve osservazione
  durante il lavoro.
- Richiedere prova coerente di layer offloaded nella diagnostica nativa
  (contatore **maggiore di zero**), identificativo GPU corrispondente,
  generazione e ritorno di token reali. Se log o telemetria mancanti,
  registrare `inconclusive`, non `verified`.
- Registrare esito modello, contesto, tempi di load/prefill/decode,
  token generati, eventuali OOM, stderr ridotto e metriche VRAM.
  Evitare prompt con dati privati e non registrare il contenuto del
  modello come log. Output diagnostico strutturato JSON.
- In caso di crash, OOM, timeout o assenza di segnali GPU, interrompere
  e lasciare il runtime integro; niente fallback silenzioso.

## Criterio di successo

Separare esplicitamente:
`runtime_loadable`, `model_loaded`,
`cuda_backend_supported`, `gpu_offload_observed`,
`tokens_generated`, `inference_executed`, `gpu_inference_verified`.

`gpu_inference_verified=true` solo dopo **inferenza reale con token**
e prova dell'uso GPU. Una `libllama.so` corretta, uno smoke CUDA
o un `llama_supports_gpu_offload()==true` non bastano.

## Confini architetturali

La C ABI diretta è già disponibile per inizializzazione/system info,
ma un engine di tokenizzazione/decoding C autonomo resta TO DO.
Per la **prima inferenza** riusare `CudaLlamaAdapter`: i metodi Python
e C usano già lo stesso file `libllama.so`, dunque non servono
ricompilazioni né un secondo runtime.

`diskcache` CVE-2025-69872 rimane TO DO non prioritario; la sua
presenza in una installazione esplicitamente autorizzata non equivale
a una correzione upstream.
