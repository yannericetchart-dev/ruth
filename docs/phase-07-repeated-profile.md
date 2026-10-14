# Ruth — Fase 07: profiling ripetibile CUDA GGUF

## Stato precedente e prerequisiti

L'utente ha confermato il test funzionale reale del GGUF
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` su RTX 4050 Laptop:
`gpu_gguf_inference_verified`, **17/17 layer offloaded**, 24 token,
940 MiB massimi osservati tramite `nvidia-smi`, durata worker 0,834 s.
La suite di 278 test precedenti è stata successivamente confermata
verbalmente dall'utente. Il precedente smoke non era un benchmark:
la durata include più fasi e la VRAM è campionata.

## Prima baseline reale verificata — 14 ottobre 2026

Sul PC Ubuntu/RTX 4050 Laptop l'utente ha eseguito
`./ruth models profile --purpose test --allow-gpu-inference`.

**Risultato:** `status=gpu_profile_verified`, `error=null`,
`gpu_profile_verified=true`, `model_inference_executed=true`,
`model_verified=true`, `native_library_verified=true`,
`gpu_process_observed=true`, `cpu_fallback_selected=false`.
**296/296 test OK** in **2,146 s**.

- **GPU:** PCI `00000000:01:00.0`, UUID
  `GPU-5a841828-882b-85d9-c98e-3a0d29e66e00`.
- **Layer sul dispositivo:** `17/17` (offload completo del modello di test).
- **VRAM massima campionata per il processo:** **956 MiB** (non un massimo
  fisico garantito e non una curva temporale).
- **Caricamento modello:** **0,596538359 s**, misurato **una volta**.
- **Protocollo:** `n_ctx=512`, `seed=42`, `max_tokens=64`, 1 warmup escluso,
  5 round misurati con 45 token di prompt e 64 token di output ciascuno
  (63 token temporizzati come decode, primo token misurato separatamente).
- **Manifest GGUF:** `Llama-3.2-1B-Instruct-Q4_K_M.gguf`,
  SHA256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`,
  revision Hugging Face `067b946cf014b7c697f3654f621d577a3e3afd1c`.
- **Report JSON conservato localmente:**
  `.ruth-build/profiles/llama-3.2-1b-q4km-20261014T095718617642Z.json`.
  Il file è nella cartella locale ignorata da Git; non dichiararlo presente
  sul server remoto.

| Metrica | Min | Mediana | Max |
|---|---:|---:|---:|
| Decode (token/s, 63 token successivi al primo) | 158,551 | **160,983** | 164,005 |
| Prefill (token/s, prompt 45 token) | 20.655,173 | **21.094,208** | 21.452,159 |
| Decode (secondi) | 0,384133 | 0,391347 | 0,397348 |
| Prefill (secondi) | 0,002098 | 0,002133 | 0,002179 |
| Sample del primo token (secondi) | 0,010568 | 0,010602 | 0,011434 |
| Tokenizzazione (secondi) | 0,000298 | 0,000302 | 0,000326 |

Velocità decode per round (token/s, dato osservato):
`[164.005494, 158.551181, 162.668360, 160.982529, 160.136227]`.
Media circa **161,269 token/s**, deviazione standard campionaria circa
**2,133 token/s** (CV circa **1,32%**) su **cinque round nella stessa sessione
calda**. Non sono cinque cold start indipendenti né un intervallo di
confidenza per altre lunghezze del contesto.

**Interpretazione e limiti:** le misure prefill sono su **soli 45 token**,
quindi particolarmente sensibili a overhead e cache CUDA; i ~21k token/s
non generalizzano a prompt lunghi. `first_token_seconds` è solo
`sample()` dai logits del prefill, **non TTFT end-to-end**.
I **956 MiB** misurano il picco campionato per processo con `nvidia-smi`,
e non sono un margine di VRAM trasferibile ai modelli di altra dimensione.
Il precedente smoke di **940 MiB** è una misura distinta con finestra di
campionamento diversa, non evidenza di regressione di memoria.
Il tempo di caricamento **0,597 s** non va sommato implicitamente ad
ogni richiesta generata in una sessione persistente.

**Conclusione:** baseline CUDA funzionale e ripetibile per il *caso test*
Llama 1B, ma non una qualificazione del modello production Gemma 2 9B IT
né dei 6 GiB in contesti prolungati. Occorrono misure su più prompt/context,
termica/power e più processi di benchmark indipendenti prima di generalizzare.

## Comando (nessuna installazione / nessun download)

```bash
./ruth models profile --purpose test --allow-gpu-inference
```

Solo modello di test Llama 3.2 1B, file e manifesto GGUF già verificati
con SHA256/revisione; C ABI/Python stessa `libllama.so` verificata prima
di ogni profilo; GPU UUID/PCI univoci e visibilità fissata con
`CUDA_VISIBLE_DEVICES`. `--allow-gpu-inference` è obbligatorio.
Gemma 2 9B IT resta la scelta di produzione, non autorizzata per un
profilo GPU finché non ci sono revisioni/pesi/asset approvati.

## Protocollo v1 (fisso per comparabilità)

- Modello: Llama 3.2 1B Instruct Q4_K_M, Hugging Face commit
  `067b946cf014b7c697f3654f621d577a3e3afd1c`.
- `n_ctx=512`, `n_batch=128`, `n_ubatch=128`;
  `n_gpu_layers=-1`, `main_gpu=0`, `seed=42`, greedy `temp=0`.
- Prompt non sensibile fisso in italiano; `max_tokens=64`.
- **1 warmup escluso** e **5 round misurati** nello **stesso worker**
  e nella **stessa sessione nativa**, stato e seed ripristinati a ogni
  round. Non confondere questo protocollo *warm* con cold start per
  ciascuna richiesta.
- `model_load_seconds`: inizializzazione binding, allocazione/caricamento
  del GGUF e posizionamento dei pesi; misurato **una volta per worker**,
  non aggregato come se ci fossero cinque caricamenti.
- `tokenize_seconds`: tokenizzazione del prompt prima di `eval`.
- `prefill_seconds`: tempo effettivo di `Llama.eval(prompt_tokens)`,
  distinto dall'inferenza del primo token.
- `first_token_seconds`: `Llama.sample(temp=0)` eseguito sui logits
  del prefill. **NON è TTFT end-to-end**.
- `decode_seconds`: loop dei token **successivi al primo**, misurando
  `Llama.eval(previous)` + `Llama.sample(next)`; almeno 8 token
  dopo il primo per considerare significativa la misura.
- `prefill_tokens_per_second = prompt_tokens / prefill_seconds`;
  `decode_tokens_per_second = decode_tokens / decode_seconds`.
  Le due grandezze hanno denominatori e significati differenti.
- Per le misure di fase vengono riportati **minimo, mediana, massimo**
  dei cinque round. Non inventare throughput dove manca il dato.
- `nvidia-smi` campiona il PID worker/GPU UUID a intervallo 200 ms;
  il **picco osservato** di VRAM in MiB non è il massimo fisico certo.
- Il log nativo deve confermare **offload completo N/N dei layer**
  sul device; la sola disponibilità delle API CUDA non basta.
- Il worker non scrive prompt/risposte in output. Rende pubbliche
  solo misure aggregate e conteggi token; non abilita diskcache.

## Risultati e conservazione

Esito di successo: `gpu_profile_verified` con
`samples=5`, `warmup_rounds=1`, `statistics` e
`measurements`. I report sono conservati come nuovi JSON non
sovrascrivibili sotto:

` .ruth-build/profiles/llama-3.2-1b-q4km-<UTC>.json `

I file includono SHA256 del modello, revisione e UUID/PCI della GPU,
opzioni fisse e misure per round. La cartella è già ignorata da Git.

Se uno dei gate non passa (hash modello, ABI, PID GPU, log dei layer,
numero di campioni, timing o token incoerenti), Ruth restituisce un
errore/inconclusivo: **non seleziona un backend CPU, non modifica la
`.venv`, non scarica modelli e non compila**.

### Limiti

Si tratta della prima **baseline controllata** su Llama 1B, non di una
caratterizzazione completa della qualità del modello né di uno stress
test sostenuto su diversi contesti. Misure sensibili a temperatura,
power limit, altre attività GPU e scheduler; confrontare più run e
registrare tali condizioni prima di generalizzare. I soli risultati
Llama 1B non qualificano **Gemma 2 9B IT** su 6 GiB VRAM.

Kaggle provider e CVE `diskcache` restano TO DO, senza modifiche.
