# Ruth — Gemma 2 9B IT Q4_K_L: pin e prima prova minima CUDA

## Decisione Phase 05 (14 ottobre 2026)

L'utente ha confermato la scelta **Gemma 2 9B IT Q4_K_L** come
prima candidata dal catalogo GGUF per produzione, alla luce della
shortlist reale calcolata da Ruth (5759 MiB liberi su RTX 4050 Laptop).

**La quantizzazione Q4_K_L è stata pin-nata nel catalogo**:
- Modello: `gemma/gemma-2/gemma-2-9B-it/gguf/Q4_K_L`
- Repository pubblico GGUF: `bartowski/gemma-2-9b-it-GGUF`
- Commit HF: `d731033f3dc4018261fd39896e50984d398b4ac5`
- Filename: `gemma-2-9b-it-Q4_K_L.gguf`
- Dimensione: **5983265728 byte** (~5706,09 MiB)
- SHA256 LFS: `ed8573f0d1e3ca58e471498fd162a65d4f4253dffdec28a7ba394fdcae92f1f3`
- Licenza: **Gemma**, NON ancora accettata per questo download.
- Allo snapshot GPU iniziale, differenza filesize/VRAM free: ~**52,91 MiB**;
  non è margine runtime misurato né garanzia che CUDA riesca a caricare.

La variante `Q4_K_M` rimane in catalogo con i propri pin
(5761057728 byte, SHA256
`13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe`)
ma **non** è il default. Questo non configura un fallback automatico:
Ruth **non passa da Q4_K_L a Q4_K_M silenziosamente** in caso di
OOM, errore di modello o problemi di driver.

## Verifica senza download né licenza

```bash
./ruth models select --purpose production
./ruth models resolve --purpose production --allow-metadata-network
```

La selezione mostra default `Q4_K_L`; la risoluzione (solo metadati)
verifica che hash, commit, dimensione e licenza coincidano con i pin.
Nessun comando carica i pesi in GPU o avvia un trasferimento di GGUF.

## Gate di licenza e trasferimento

I termini originali di Google Gemma vanno esaminati
**prima** di autorizzare il download:
https://ai.google.dev/gemma/terms

Dopo la valutazione e una dichiarazione esplicita di accettazione
dei termini da parte dell'utente, il trasferimento è una seconda
azione separata, controllata da **due flag** nel downloader esistente:

```bash
./ruth models download --purpose production --allow-model-download --accept-model-license
```

Il comando **NON** viene eseguito dalla selezione, dalle modifiche
al catalogo, dai test o dai comandi di sola lettura. Il download
controlla revisione HF, dimensione esatta, SHA256 LFS e full-file,
manifest `license_acknowledged_for_this_download`, lock e resume.
Prima della prova nativa `model_verified=true` deve essere
confermato sul PC.

## Prova minima n_ctx = 128 (implementazione pronta, non eseguita)

Una volta che l'utente ha autorizzato il download e ottenuto il GGUF
verificato, il comando successivo richiede separata autorizzazione
all'inferenza GPU:

```bash
./ruth models probe-production --allow-gpu-inference
```

- Nessun download, nessuna installazione o build, nessuna licenza
  accettata implicitamente.
- Verifica file GGUF completo e `*.manifest.json` con SHA256/revision
  esatti **prima** di importare `llama_cpp` nel worker.
- Controlla .venv Python 3.12 e provenienza della libreria native CUDA.
- Worker separato con `CUDA_VISIBLE_DEVICES` legato a una sola GPU UUID,
  `n_gpu_layers=-1`, `main_gpu=0`, `n_ctx=128`,
  `n_batch=64`, `n_ubatch=64`, `seed=42`, prompt breve
  e massimo 24 token. Nessuna cache disco o fallback CPU.
- Telemetria `nvidia-smi` per PID/UUID/VRAM massima **campionata**.
- Parsing log nativo: **offload completo N/N** richiesto per esito OK.
- Richiede modello caricato, token realmente generati e processo visto
  dalla telemetria GPU per `gemma_min_context_gpu_verified`.
- Se GPU non sufficiente: errore/worker failed/inconclusivo, NON
  `gpu_inference_verified=true`, NON selezione di una quantizzazione
  peggiore, NON `runtime_profile` di produzione.
- `context_calibration_pending=true`, `runtime_profile_published=false`
  anche in caso di esito positivo.

La soglia `n_ctx=128` è il **primo esperimento minimo stabilito**
e **non** un massimo operativo qualificato. La Fase 09 verificherà
progressivamente allocazione reale/KV/compute, latenza, stabilità,
contesto massimo e margine osservato. In accordo con la scelta
dell'utente, Phase 05 non detrae riserve di VRAM.

Non confondere la VRAM del file GGUF, il consumo del processo GPU,
l'offload layer riportato nei log e una vera garanzia di
residenza integrale di tutti i tensori. La prima prova richiede
evidenza indipendente (log GPU + PID/VRAM + token), ma non costituisce
ancora un benchmark né una qualifica production completa.
