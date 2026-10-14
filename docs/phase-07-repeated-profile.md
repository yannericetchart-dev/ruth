# Ruth — Fase 07: profiling ripetibile CUDA GGUF

## Stato precedente e prerequisiti

L'utente ha confermato il test funzionale reale del GGUF
`Llama-3.2-1B-Instruct-Q4_K_M.gguf` su RTX 4050 Laptop:
`gpu_gguf_inference_verified`, **17/17 layer offloaded**, 24 token,
940 MiB massimi osservati tramite `nvidia-smi`, durata worker 0,834 s.
La suite di 278 test precedenti è stata successivamente confermata
verbalmente dall'utente. Il precedente smoke non era un benchmark:
la durata include più fasi e la VRAM è campionata.

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
