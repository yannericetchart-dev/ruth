# Ruth — Diagnostica del primo caricamento Gemma Q4_K_L (Fase 09)

## Aggiornamento: CUDA OOM dimostrato dai log nativi (14 ottobre 2026)

Dopo il rilascio della diagnostica, la ripetizione del probe su RTX
4050 ha confermato la causa: `cudaMalloc failed: out of memory`
mentre il runtime tentava `allocating 5700.31 MiB on device 0`.
Risultato `native_failure_category=cuda_out_of_memory_reported`,
segnali `explicit_out_of_memory`, `cuda_allocation_failed`,
`model_load_failed_generic`. GGUF/hash e ABI nativa verificati,
ma `model_loaded=false` e **nessun token generato**.

La diagnosi sotto riportata documenta il primo tentativo, nel
quale lo stderr non era ancora esposto. **Per lo stato corrente
fa fede l'OOM confermato**, descritto con il prossimo candidato in
`docs/phase-09-gemma-oom-next-candidate.md`. Non ripetere senza motivo
Q4_K_L; proseguire solo con Q4_K_M esplicitamente selezionata,
scaricata e verificata con le consuete autorizzazioni separate.



## Esito realmente osservato — 14 ottobre 2026

Comando eseguito sul PC:

```bash
./ruth models probe-production --allow-gpu-inference
```

Output rilevante:

- `status=production_min_context_worker_failed`
- `error=ValueError: Failed to load model from file: ...Q4_K_L.gguf`
- `model_verified=true`: manifesto e SHA256 GGUF controllati localmente
- `native_library_verified=true`: binding/libllama CUDA verificati
- `n_ctx=128`, `n_batch=64`, `n_ubatch=64`
- `model_loaded=false`, `generated_tokens=0`, `offloaded_layers=0`
- `gpu_process_observed=false`, `gpu_memory_peak_mib_observed=null`
- `cpu_fallback_selected=false`, `runtime_profile_published=false`

**Il caricamento fallisce, ma non è stato ancora diagnosticato come
OOM.** Il `ValueError` generico non riporta il motivo emesso dalle
librerie native. La prima implementazione teneva `verbose=True` su
llama.cpp, ma catturava e scartava `stderr` quando il worker restituiva
l'errore strutturato.

Dai metadati reali della prima selezione, file GGUF Q4_K_L
`5983265728 byte` (~5706,09 MiB) contro ultimo snapshot
`5759 MiB` di memoria NVIDIA libera: differenza **~52,9 MiB**.
È un **forte indizio di saturazione** ma non una prova di CUDA OOM;
il margine istantaneo può cambiare. Il filesize GGUF non equivale a
residenza tensor completa in VRAM: solo la prova nativa può accertarla.

## Correzione v2.30 — conservazione diagnostica sicura

Un nuovo modulo `ruth_runtime/native_failure_diagnostics.py`
legge esclusivamente `stderr` del processo isolato e restituisce:

- `native_failure_category`: `cuda_out_of_memory_reported` **solo**
  in presenza di segnale nativo esplicito; categorie distinte per
  allocazione CUDA fallita, errore driver/runtime, tensor/GGUF
  non supportato o causa ignota.
- `native_failure_signals`: categorie basate su espressioni
  effettivamente presenti, non sulla dimensione del file.
- `native_log_excerpt`: al massimo otto righe pertinenti
  con percorsi utente e URL oscurati; NON è un dump indiscriminato
  dello stderr e non stampa prompt/risposta.
- `native_cuda_buffer_mib_reported`: dimensione massima del buffer
  CUDA registrato da llama.cpp, se il logger fornisce la misura.
- `worker_exit_code` per distinguere eccezioni regolari e crash.
- `generic_python_load_error_only`: il solo `Failed to load model`
  non basta ad attribuire una causa GPU.

La diagnostica opera anche se il processo fallisce senza il risultato
strutturato JSON. I campi nuovi non cambiano né il risultato positivo
né i gate di verifica (integrità, consenso esplicito, worker isolato,
offload completo, PID GPU, token reali).

## Riprovare dopo l'aggiornamento del codice

```bash
git pull --ff-only
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
./ruth models probe-production --allow-gpu-inference
```

Attesi **348 test**, ancora da confermare sul PC dopo il commit.
Il probe ripete **solo** la prova già autorizzata su modello locale:
niente nuovo download, accettazione di licenza, `.venv`, build o
commutazione della quantizzazione.

Per affiancare una fotografia della VRAM libera:

```bash
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits
```

Questo snapshot **non** viene sottratto dalla Fase 05: la selezione
continua a usare tutta la VRAM libera, senza riserva.

## Decisione successiva, guidata dai risultati

1. Se i log attestano `cuda_out_of_memory_reported` o un fallimento
   di allocazione, classificare Q4_K_L come non caricabile con
   **offload completo** nelle condizioni reali misurate a n_ctx=128.
   Non correggere artificialmente il dato di memoria né attivare CPU.
2. Considerare le quantizzazioni successive già catalogate,
   inizialmente **Q4_K_M** (non scaricata automaticamente) oppure una
   variante più piccola **IQ4_XS**, se scelta e pin-nata
   successivamente. Per ogni GGUF serve separata autorizzazione
   trasferimento e verifica hash/licenza.
3. Se emergono incompatibilità quantizzazione, GGUF o errore driver,
   non concludere OOM; correggere solo il problema dimostrato.
4. Se le evidenze restano insufficienti, riportare `cause_unknown`
   e approfondire senza promuovere un profilo production.

Solo una nuova prova con full offload e generazione verificata può
aprire la calibrazione progressiva `n_ctx` della Fase 09.
