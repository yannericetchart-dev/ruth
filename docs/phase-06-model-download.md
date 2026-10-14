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
   siano pinnati. Gemma 2 9B IT Q4_K_L ha pin completi, ma **non viene scaricato** senza
   autorizzazione distinta per i termini Gemma e il trasferimento.
2. Rilegge in sola modalità metadata le informazioni di Hugging Face,
   richiedendo identità immutabile, LFS SHA256, bytes e licenza coerenti.
   Un repository gated non viene aperto implicitamente.
3. Controlla filesystem, symlink, download concorrenti e spazio disponibile
   (byte mancanti + almeno 128 MiB).
4. Scarica dalla risorsa
   `https://huggingface.co/{repo}/resolve/{commit}/{filename}?download=true`;
   sono ammessi **solo HTTPS e host Hugging Face/CDN/Xet esplicitamente
   elencati**. Non invia credenziali a provider esterni.
### Redirect CDN Hugging Face (fix dopo primo download)

Il primo tentativo reale di download del modello Llama 1B è stato fermato
prima di ricevere byte: `ValueError: Model redirect outside allowlisted HTTPS
Hugging Face CDN`. Il file `*.part` risultava vuoto (`bytes_present=0`);
questo **non** costituisce fallimento del checksum o incompatibilità GGUF.

La lista iniziale di hostname era incompleta. L'elenco esplicito nel downloader
ora include i CDN regionali documentati da Hugging Face:

- `us.aws.cdn.hf.co`, `us.gcp.cdn.hf.co`
- `cdn-lfs-us-1.hf.co`, `cdn-lfs-eu-1.hf.co`
- `cas-server.xethub.hf.co`, `cas-server.xethub-eu.hf.co`
- `transfer.xethub.hf.co`, `transfer.xethub-eu.hf.co`

Sono conservati i precedenti host Xet/LFS specifici. Non si abilita
un wildcard `*.hf.co`, non si autorizzano indirizzi esterni, HTTP,
credenziali embedded negli URL o porte inattese. I redirect vengono
verificati e la destinazione finale è ricontrollata. I messaggi di errore
mostrano **solo il nome host** e non pubblicano query firmate.

**Fonte ufficiale:** https://huggingface.co/docs/hub/models-downloading
(sezione download dietro proxy/firewall, elenco host CDN).

Dopo `git pull --ff-only` e i test, è possibile ripetere lo stesso comando
di download già autorizzato **solo se l'utente intende nuovamente avviare
il trasferimento**, sempre con entrambi i flag espliciti. La directory
parziale da zero byte viene riusata in sicurezza. Una differente
destinazione CDN non comporta mai la riduzione delle verifiche di SHA256,
dimensione e revisione Git; se compare un altro host, Ruth fallisce e ne
riporta il solo hostname per verifica prima di modificare la policy.

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

`gemma-2-9B-it` resta default production con **Q4_K_L pinnata**
(5983265728 byte), ma **non è ancora qualificata** per 6 GiB di VRAM.
La Fase 05 non riserva VRAM; la prova nativa CUDA e il contesto
saranno calibrati iniziando da n_ctx=128, dopo il download autorizzato.

La CVE `diskcache` resta TO DO non prioritario e `LlamaDiskCache`
rimane disabilitata. Il provider Kaggle resta TO DO.
