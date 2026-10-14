# Ruth — Prima inferenza GGUF CUDA reale (Fase 07)

## Prerequisiti verificati sul PC

Il primo download Hugging Face del GGUF di test è concluso con
`model_download_verified`, `model_verified=true`, `bytes_present=807694464`,
SHA-256:
`6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`.

Modello: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, revisione immutabile
`067b946cf014b7c697f3654f621d577a3e3afd1c`.
La licenza Llama 3.2 è stata esplicitamente dichiarata accettata per il
download. Le 15 librerie native sono già qualificate e il binding CUDA
è installato nella sola `.venv`; nessuna nuova build o installazione serve.

## Comando (solo dopo test)

```bash
./ruth models smoke --purpose test --allow-gpu-inference
```

Senza `--allow-gpu-inference` il comando non carica pesi e non avvia
subprocessi GPU. Questa autorizzazione **non** scarica ulteriori modelli.

Il comando è una prova controllata (un solo prompt fisso non sensibile,
`n_ctx=512`, `n_batch=128`, `n_ubatch=128`, massimo 24 token), non
un servizio chat. Rifiuta `--purpose production` perché Gemma 9B non
è ancora qualificato e la VRAM disponibile non ne garantisce l'offload
completo.

## Verifiche e raccolta delle evidenze

1. Ricontrolla il GGUF finale e manifest `*.manifest.json` (revision
   Git, SHA256, dimensione, license-ack, marker GGUF); non usa parziali
   `*.part`, URL o contenuti non verificati.
2. Verifica ancora tramite subprocessi che la libreria CUDA installata
   abbia provenienza pinned, ABI diretta e Python coincidenti.
3. Richiede `nvidia-smi` e **una sola NVIDIA GPU identificabile**,
   con UUID, bus PCI e almeno 512 MiB liberi. Se più GPU sono visibili,
   blocca invece di scegliere a caso.
4. Lancia un nuovo processo con `.venv/bin/python -I` e
   `CUDA_VISIBLE_DEVICES=GPU-<UUID>`, evitando import di sistema e
   cambiamento di librerie. Usa `CudaLlamaAdapter` con
   `n_gpu_layers=-1`, `main_gpu=0`, nessuna cache; verbose native
   diagnostics abilitate solo per il test.
5. Campiona in parallelo il PID del worker nella telemetria
   `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory` e
   registra il massimo *osservato* di memoria dedicata al processo.
   La misura è una singola prova, **non** un benchmark comparabile.
6. Esegue una generazione reale via `create_completion`,
   raccogliendo `usage.completion_tokens` dall'API e SHA256 della risposta,
   senza stampare il testo prodotto.
7. Cattura i diagnostici nativi di llama.cpp e richiede una voce
   `offloaded N/M layers to GPU` con **N maggiore di zero**.
8. Dichiara `gpu_gguf_inference_verified` solo se nello stesso worker
   il modello è caricato, vengono generati token (`completion_tokens>0`),
   l'offload di layer è documentato dal log e il PID del processo viene
   visto consumare memoria sulla **GPU UUID selezionata**.

Se il worker genera token ma manca un elemento di prova GPU, l'esito è
`gpu_inference_inconclusive`. In caso di eccezioni, timeout,
incompatibilità modello/pin o backend senza supporto, viene restituito
un errore esplicito. **Mai** un fallback automatico CPU o altra build.

Il comando non installa pacchetti, non cambia driver, non modifica il
wheel o la `.venv`, non scarica pesi, non pubblica profili operativi e
non imposta un backend di produzione. Un `llama_supports_gpu_offload()`
true da solo non è una prova di inferenza GPU.

## Limitazioni e prossimi passaggi

- Il logger nativo può variare formato nelle future versioni; in tal
  caso si aggiorna il parser dopo aver letto evidenze affidabili,
  **senza inventare il numero di layer offloaded**.
- Campioni `nvidia-smi` con intervallo 200 ms non garantiscono il
  massimo assoluto VRAM e non sostituiscono strumenti di profiling.
- Lo schema `ruth-cuda-gguf-smoke-v1` è un gate funzionale iniziale,
  non il `runtime_profile` ottimizzato.
- Test C ABI di generazione diretta restano TO DO separato: l'adapter C
  è già presente per diagnostica/backend, ma l'inferenza qui usa il
  binding Python **sulla stessa libllama.so**.
- Gemma-2-9B-it resta default production, ma non è stato misurato e
  non può essere dichiarato compatibile con soli 6GiB di VRAM.
- Kaggle e la CVE diskcache rimangono rispettivamente TO DO provider
  e TO DO sicurezza, senza modifiche.
