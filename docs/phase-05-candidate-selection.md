# Ruth — Fase 05: selezione quantizzazioni con **tutta la VRAM libera**

## Decisione approvata

**Nessuna riserva di VRAM durante la selezione.** Ruth sceglie la
quantizzazione migliore in termini di `quality_priority_heuristic`
fra quelle il cui **file GGUF esatto in byte** non supera la
**VRAM libera riportata da NVIDIA** al momento della ricerca.

```
file_gguf_size_bytes <= nvidia_free_mib * 1048576
```

La Fase 05 **non** sottrae memorie arbitrarie per
runtime, kernel CUDA, buffer computazionali, KV cache, cache allocator
o margini di sicurezza. Non prende decisioni con una formula
di riserva fissa. La quantità risultante è la dimensione del file
su disco contro la memoria libera istantanea, *non* una dimostrazione
di caricabilità CUDA: non confondere una prova dimensionale con
l'allocazione reale.

## Gerarchia e discovery

- Provider predefinito: **Hugging Face**; **Kaggle TO DO**.
- Famiglie registrate: **Gemma, Llama, Qwen, Mistral**.
- Percorso: `family/model/version/gguf/quantization`.
- Default production: **Gemma 2 9B IT** (quantizzazione da selezionare).
- Default test: **Llama 3.2 1B Instruct Q4_K_M**, già verificato
  realmente su RTX 4050.

```bash
./ruth models quantizations --purpose production --allow-metadata-network
./ruth models recommend --purpose production --allow-metadata-network
```

`quantizations` legge metadata HF (revision Git immutable, LFS
SHA256, dimensione intera per file). `recommend` in più legge
`nvidia-smi --query-gpu=uuid,pci.bus_id,memory.total,memory.free`:
accetta solo una GPU identificabile, ordina le candidate per qualità
euristica e presenta quelle entro **tutta** la VRAM libera.

**Non esiste più `--reserve-mib`** e non viene usata alcuna
riserva implicita. Il limite di shortlist rimane personalizzabile
con `--top-k` (da 1 a 10; default 5).

Risultati:
- `size_budget_mib` = memoria GPU *libera* (senza detrazioni).
- `size_budget_bytes` = `free_mib * 1048576`.
- `size_only_prefilter=pass_requires_native_test` per una candidata.
- `estimated_remaining_mib_after_file` = differenza teorica fra
  VRAM libera e dimensione del GGUF; può essere molto piccola o zero.
- `first_candidate_quantization` = variante migliore secondo il
  ranking **qualitativo euristico** fra quelle che passano.
- `candidate_is_production_qualified=false` e
  `full_gpu_residency_verified=false` fino ai test CUDA reali.
- `context_calibration_pending=true` e
  `runtime_memory_calibration_pending=true`.

La quality policy versionata, in
`catalog/quantization_policy.json`, usa un ordine di preferenza
non misurato. Differenti algoritmi (ad esempio K-quant e IQ-quant)
richiederanno confronti reali a parità di modello e workload.
I GGUF speciali ARM sono esclusi dalla scelta CUDA.

## Contesto minimo: inizio della calibrazione, NON filtro

Il candidato selezionato è proposto con
**`initial_probe_n_ctx=128`**. È un *contesto iniziale piccolo e
dichiarato*, non un minimo architetturale garantito per qualunque modello.
Il valore non influenza la selezione del file, né implica un consumo KV
stimato. Nel primo futuro worker di produzione:

1. Usare questo contesto piccolo e un prompt/test output che vi entri.
2. Caricare il GGUF in un **processo CUDA isolato**, chiedendo offload
   completo e senza fallback CPU.
3. Misurare davvero model load, allocazioni CUDA/KV e generazione.
4. In **Fase 09**, ricercare `n_ctx` maggiore con prove ripetute,
   misurando VRAM, offload, throughput e affidabilità.
5. Se il candidato fallisce all'avvio con contesto minimo, registrare
   errore/OOM senza classificare falsamente un successo e valutare
   la prossima quantizzazione; non introdurre riserve arbitrarie
   retroattivamente nella Fase 05.

La fase di selezione non carica pesi, non usa CPU, non cambia la
`.venv` e non autorizza alcun download o accettazione di licenza.

## Dati reali del caso Gemma 2 9B / RTX 4050

Dalla verifica HF eseguita sul PC:
- Variante GGUF `Q4_K_M`, SHA256
  `13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe`.
- Revision Git
  `d731033f3dc4018261fd39896e50984d398b4ac5`.
- File esatto: **5761057728 byte ≈ 5494,17 MiB**.
- NVIDIA RTX 4050 Laptop: **6141 MiB totali**, **5759 MiB liberi**
  nello snapshot riportato.

**Q4_K_M passa il prefiltro** (5494,17 MiB <= 5759 MiB)
e, secondo la policy attuale, è la priorità più alta fra i GGUF
che rientrano nello stesso snapshot. La differenza teorica è circa
**264,83 MiB** e **non** rappresenta memoria runtime garantita.
Se il caricamento reale non riesce, questa quantizzazione **non** sarà
qualificata per production; la calibrazione potrà valutarne altre.

## Gate successivo (esplicito, non automatico)

Dopo aver letto la shortlist reale:
1. Scegliere e pinning in catalogo di quantizzazione, filename,
   SHA256, commit e dimensione verificati; nessuna mutazione da
   `recommend`.
2. Leggere e accettare la licenza originale Gemma e autorizzare
   separatamente il download (nessun download attualmente avviato).
3. Provare il contesto iniziale piccolo con worker CUDA isolato.
4. Calibrare contesto, VRAM e performance usando dati osservati.
5. Pubblicare un profilo production **solo** dopo verifica effettiva.

Il modello Llama 1B rimane la baseline verificata; la CVE
`diskcache` rimane TO DO non prioritario con cache disabilitata.
