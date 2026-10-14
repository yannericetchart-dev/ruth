# Ruth — Phase 09: Gemma Q4_K_L CUDA OOM e prova esplicita Q4_K_M

## Risultato reale — 14 ottobre 2026

La prima prova production sulla RTX 4050 Laptop 6 GiB è stata eseguita
con `./ruth models probe-production --allow-gpu-inference`, Gemma 2 9B
IT `Q4_K_L`, `n_ctx=128`, `n_batch=64`, `n_ubatch=64`.

**Risultato:** `production_min_context_worker_failed`,
`native_failure_category=cuda_out_of_memory_reported`, segnale nativo
`cudaMalloc failed: out of memory`, exit code 2. I diagnostici nativi:

```text
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5700.31 MiB on device 0: cudaMalloc failed: out of memory
llama_model_load: error loading model: unable to allocate CUDA0 buffer
llama_model_load_from_file_impl: failed to load model
```

Il file GGUF e il runtime native CUDA sono stati verificati, ma
`model_loaded=false`, `generated_tokens=0`, nessun layer offloaded
attestato, memoria PID non disponibile e `gpu_inference_verified=false`.
Snapshot successivo `nvidia-smi`: RTX 4050 Laptop,
UUID `GPU-5a841828-882b-85d9-c98e-3a0d29e66e00`,
**6141 MiB totali**, **5759 MiB liberi**.
Il buffer CUDA richiesto era **5700.31 MiB**; il gap aritmetico con
lo snapshot successivo è 58,69 MiB. Lo snapshot NON è una misura della
memoria libera al microistante del fallimento e non è una prova che
liberare precisamente 58,69 MiB sia sufficiente.

**Conclusione limitata ma definitiva:** sulla configurazione reale
osservata, Q4_K_L non è riuscito a caricare tutti i pesi in GPU nemmeno
a contesto iniziale 128. Non confondere questo con la dimostrazione di
incompatibilità universale del GGUF o di tutti i device CUDA.

## Conduzione della calibrazione, senza riserve VRAM

La Fase 05 resta invariata: `file GGUF size <= VRAM libera` e
ordinamento euristico per qualità, **nessuna riserva**.
La Fase 09 conserva i fallimenti reali e passa *esplicitamente* alla
variante successiva nel catalogo già pinned, senza fallback automatico.

**Prossimo candidato:** Gemma 2 9B IT `Q4_K_M`, stessa revisione immutabile
`d731033f3dc4018261fd39896e50984d398b4ac5`,
dimensione **5761057728 byte** (~5494,17 MiB),
SHA256 `13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe`.

La scelta di Q4_K_M non garantisce che il runtime CUDA lo possa
allocare. Il probabile vantaggio dimensionale è una ipotesi da
testare, non un dato osservato di VRAM residua.

## Gate e comandi

Aggiornare il codice, quindi verificare i test **senza accesso alla rete**:

```bash
git pull --ff-only
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

**356 test attesi**, nuova suite non ancora eseguita sul PC.

Controllo in sola lettura del pin della variante alternativa:

```bash
./ruth models resolve --purpose production --quantization Q4_K_M --allow-metadata-network
```

Nessuna chiamata al downloader è implicita. Se il file Q4_K_M non è
già presente, il nuovo probe deve rispondere con
`model_integrity_or_license_unverified` prima di usare CUDA.
Per trasferire i pesi nuovi serve **nuova azione esplicita** e
dichiarazione di licenza originale Gemma accettata:

```bash
./ruth models download --purpose production --quantization Q4_K_M --allow-model-download --accept-model-license
```

Il flag `--accept-model-license` significa che l'utente dichiara
di aver letto e accettato i termini Google Gemma:
https://ai.google.dev/gemma/terms.
Non eseguire il trasferimento sulla macchina altrui senza tale
autorizzazione; il catalogo da solo non costituisce consenso.

Solo con `model_download_verified` e autorizzazione distinta al
caricamento GPU:

```bash
./ruth models probe-production --purpose production --quantization Q4_K_M --allow-gpu-inference
```

Il probe ora inoltra la quantizzazione al processo figlio e richiede
concordanza tra richiesta, scelta catalogo, path GGUF e report.
`Q4_K_L` resta il default pinned e il risultato OOM è conservato
nell'assessment hardware-specifico. **Nessun cambio automatico** di
default, download, CPU offload, contesto o modello durante il test.

## Come interpretare i prossimi risultati

- `gemma_min_context_gpu_verified`: prova funzionale con full
  GPU offload N/N + token generati + PID e VRAM del device; consente
  di iniziare la calibrazione progressiva di `n_ctx`.
- `production_min_context_worker_failed` con OOM/alloc failure:
  il nuovo candidato non è qualificato nelle condizioni del test.
  Scegliere esplicitamente una quantizzazione più compatta con
  pin/sorgente/provenienza confermati; non pubblicare un profilo
  falsamente riuscito.
- `model_integrity_or_license_unverified`: manca o non coincide il
  GGUF/manifest, **non** significa OOM.
- Altre categorie native: non attribuire a memoria un'incompatibilità
  di GGUF, driver, tensori o backend.

La versione aggiornata di `native_failure_diagnostics.py` estrae
anche il valore `allocating 5700.31 MiB on device 0` nel campo
`native_cuda_buffer_mib_reported` (oltre alle vecchie righe
`CUDA0 buffer size = X MiB`) **solo se il logger lo fornisce**.

Kaggle rimane TODO e la CVE `diskcache` resta aperta con cache
disco non abilitata.
