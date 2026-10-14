# Ruth — Fase 08: prossimi benchmark e fattibilità Gemma 2 9B IT

**Stato:** piano; nessun nuovo comando di benchmark implementato in questa
revisione. La baseline warm Llama 1B CUDA è stata **verificata realmente**
sul PC dell'utente (vedere `docs/phase-07-repeated-profile.md`).

## Due obiettivi distinti

1. **Riproducibilità Llama 1B e carichi rappresentativi:** la prima misura
   completa è 1 warmup + 5 round, *stessa sessione*, prompt 45 token,
   output 64 token, contesto 512, mediana decode 160,983 tok/s, VRAM
   processo campionata 956 MiB. Prima di ottimizzare il runtime, servono
   più processi indipendenti e lunghezze di prompt/contesto variabili.
2. **Fattibilità Gemma 2 9B IT production:** il default rimane
   `gemma/gemma-2/gemma-2-9B-it/gguf/Q4_K_M`, ma il file è stimato
   intorno a 5,76 GB contro una RTX 4050 Laptop da 6 GiB. Dimensione
   GGUF non equivale a occupazione VRAM: contano tensor allocation,
   KV cache, buffer computazionali, memoria già allocata e contesto.
   Non è dimostrato l'offload completo né la sua impossibilità.

## Prima azione read-only: metadata e inventario

Questi comandi **non scaricano pesi**, non caricano modelli e non
accettano licenze:

```bash
./ruth models resolve --purpose production --allow-metadata-network

nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free \
  --format=csv,noheader,nounits
```

Il resolver HF confronta lo SHA256 LFS remoto con il pin già presente
nel catalogo; se differisce, dichiara `source_metadata_unverified`
senza riparare silenziosamente il pin. Raccogliere commit immutabile,
bytes esatti, licenza e attributo `gated` **prima** di progettare il
download e stimare la pressione sulla memoria. I numeri di
`nvidia-smi` sono uno snapshot, non il consumo durante inferenza.

## Benchmark operativo successivo (da implementare)

Per Llama 1B:

- Ripetere almeno tre **processi completi distinti**, registrando warmup,
  load, prefill, decode e intervallo tra test, non solo 5 loop caldi.
- Profilare prompt di lunghezze definite e contesti crescenti
  (es. 512/1024/2048), rispettando il limite `prompt_tokens + output
  <= n_ctx`. **Non** confrontare token/s di prompt diversi come se
  misurassero lo stesso lavoro.
- Misurare TTFT completo (dall'input al primo token, distinto dalla
  sola `sample()`), latenza di decode, tokens/s sostenuti e
  VRAM/utilizzo/temperatura/potenza in funzione del tempo.
- Conservare revisioni delle librerie native, modello GGUF, quantizzazione,
  UUID GPU, clock/power osservati e parametri di inferenza.
- Valutare mediana/min/max e variabilità *tra processi*, tempi di carico
  a freddo rispetto a riuso modello, eventuali errori/OOM.
- Solo dopo questo gate confrontare candidate quantization/offload
  di Gemma 9B; nessuna sostituzione automatica di modello o fallback
  CPU durante un test che richieda GPU.

## Criteri di avanzamento

- `Gemma` rimane default *logico* di produzione, non profilo hardware
  pubblicato fino a prova di **modello reale**, generazione token,
  offload esplicito, VRAM, throughput sostenibile e stabilità.
- Qualunque modifica al catalogo/pin/quantizzazione, accettazione della
  licenza originale Gemma, o download richiede passaggi separati.
- La CVE `diskcache` resta TO DO non prioritario con cache disco
  disabilitata; Kaggle resta TO DO per il provider.

Questo piano non afferma che il modello production sia compatibile,
né che il test Llama 1B misuri le prestazioni future di Gemma.
