# Fase 06 — Download GGUF verificato e riprendibile

## Risultati verificati dall'utente (Hugging Face, modello test)

Il modello di test selezionato è:
`llama/llama-3.2/llama-3.2-1B-instruct/gguf/Q4_K_M`

Il comando `./ruth models resolve --purpose test --allow-metadata-network`
ha restituito:

- Fonte: `bartowski/Llama-3.2-1B-Instruct-GGUF` (quantizzazione di terzi)
- File: `Llama-3.2-1B-Instruct-Q4_K_M.gguf`
- Commit immutabile: `067b946cf014b7c697f3654f621d577a3e3afd1c`
- SHA256 LFS: `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`
- Dimensione: `807694464` byte
- `gated=false`, licenza modello originale `llama3.2`
- `license_accepted=false`, `download_started=false` al momento della verifica

Il commit, la dimensione e l'hash ora sono **fissati nel catalogo locale**.
Se i metadati del repository cambiano, Ruth blocca il trasferimento anziché
accettare automaticamente nuovi contenuti.

## Licenza e autorizzazione: due decisioni distinte

La quantizzazione GGUF pubblica NON elimina i termini di Meta Llama 3.2.
Leggere i termini applicabili prima di eseguire il comando:

- Repository originale: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- File GGUF pubblico: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF

Il comando richiede **entrambe** le opzioni:

```bash
./ruth models download --purpose test --allow-model-download --accept-model-license
```

`--allow-model-download` autorizza il trasferimento rete del file pin;
`--accept-model-license` è una **dichiarazione esplicita dell'utente**
di aver letto e accettato i termini applicabili. Né la selezione del
modello, né la consultazione dei metadati, né questa documentazione
costituiscono accettazione automatica.

Questo comando NON modifica la `.venv`, non esegue build, non cambia il
binding CUDA, non carica modelli e non invoca inferenza.

## Download e integrità

Dopo i due consensi il modulo `ruth_runtime/model_download.py`:
1. Verifica che catalogo, commit Git, SHA256, nome GGUF e dimensione
   siano pinnati. Il Gemma 9B predefinito rimane **non scaricabile**,
   perché non ha ancora commit e dimensione esatta verificati.
2. Rilegge in sola modalità metadata le informazioni di Hugging Face,
   richiedendo identità immutabile, LFS SHA256, bytes e licenza coerenti.
   Un repository gated non viene aperto implicitamente.
3. Controlla filesystem, symlink, download concorrenti e spazio disponibile
   (byte mancanti + almeno 128 MiB).
4. Scarica dalla risorsa
   `https://huggingface.co/{repo}/resolve/{commit}/{filename}?download=true`;
   sono ammessi **solo HTTPS e host Hugging Face/CDN/Xet esplicitamente
   elencati**. Non invia credenziali a provider esterni.
5. Scrive `*.gguf.part` e relativo `*.part.json`. Per riprendere da
   un'interruzione invia `Range: bytes=N-` e richiede
   `HTTP 206` + `Content-Range` e dimensione dei byte rigorosi.
   Se il server ignora Range, non appende byte sbagliati.
6. Confronta lunghezza totale, magic bytes `GGUF` e SHA256 di **tutto**
   il file. Soltanto dopo sposta il file al nome finale nella gerarchia
   `models/family/model/version/format/quantization/`.
7. Scrive un manifest verificabile con modello, licenza, revision,
   SHA256 e dimensione. Il GGUF resta ignorato da Git e nessun modello
   viene caricato automaticamente.

Esito di successo: `model_download_verified`, `model_verified=true`,
`model_loaded=false`, `gpu_offload_executed=false`,
`inference_executed=false`.

Altri stati: `model_download_not_authorized`,
`model_license_acknowledgement_required`, `model_source_not_verified`,
`model_insufficient_disk`, `model_download_interrupted_or_blocked`,
`model_checksum_mismatch`, `model_already_verified`.
Un `.part` incompleto resta in posizione per la ripresa, mai visibile
come GGUF verificato. Non cancellare `.part` o il suo manifest durante
un trasferimento attivo.

## Gate successivo

Il download verificato NON dimostra ancora compatibilità con GPU/VRAM,
layer offloaded o generazione di token. Dopo l'esito reale del
trasferimento sulla macchina dell'utente, il worker d'inferenza isolato
caricherà il modello di test nella stessa `.venv`, manterrà
`n_gpu_layers=-1` ed esigerà misure oggettive di offload e token
generati. Nessun fallback CPU automatico.

`gemma-2-9B-it` resta default di produzione, ma il file Q4_K_M di
circa 5,76 GB **non è ancora qualificato** per 6 GiB di VRAM
(con KV cache e buffer inclusi). Non scaricarlo né dichiararlo
compatibile sulla base del solo catalogo.

La CVE `diskcache` resta TO DO non prioritario e `LlamaDiskCache`
rimane disabilitata. Il provider Kaggle resta TO DO.
