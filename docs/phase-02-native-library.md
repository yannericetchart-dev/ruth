# Ruth — Verifica della libreria nativa condivisa (Fase 02)

## Risultato già verificato

L'utente ha installato con successo i wheel nella .venv:

- `status=native_cuda_binding_installed`
- `gpu_offload_supported=true`
- `installed_in_ruth_venv=true`
- `wheel_sha256=ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`
- `gpu_offload_executed=false` e `inference_executed=false`

**Non ricompilare e non reinstallare**: il wheel `llama-cpp-python 0.3.36`
già comprende librerie `libllama.so` e `libggml-cuda.so` compilate
con CUDA 13.2 per la RTX 4050 / sm_89. La differenza da verificare
ora non è una seconda compilazione, ma l'accessibilità diretta
dell'ABI C in aggiunta al binding Python.

## Verifica offline e read-only

Comando:

```bash
./ruth setup --verify-native-library
```

Verifica che la directory installata e l'unico file
`zz_ruth_native_cuda.pth` puntino all'installazione attesa. Ricalcola
l'hash del wheel originario e confronta **tutti i file `.so`** installati
con i byte delle librerie nel wheel pinned. Non accetta symlink o
artefatti aggiuntivi inattesi.

Poi usa due subprocessi indipendenti dell'interprete
`.venv/bin/python`:

1. **Diretto (solo standard library, `-I -S`):** `ctypes.CDLL` carica
   `libllama.so`, controlla le funzioni della pinned ABI
   `llama_supports_gpu_offload`, `llama_backend_init`,
   `llama_backend_free` e `llama_print_system_info`; inizializza
   e libera il backend senza caricare alcun modello.
2. **Python (`-I`):** importa `llama-cpp-python==0.3.36`, controlla
   `llama_supports_gpu_offload()` e confronta il percorso della sua
   libreria C nativa con **la stessa esatta `libllama.so`** usata nel
   primo processo. `LLAMA_CPP_LIB_PATH` viene eliminata dall'ambiente
   dei probe, così un override utente non può alterare il risultato.

Un esito positivo produce:

- `status=native_library_dual_abi_verified`
- `shared_libraries_verified=true`
- `direct_abi_verified=true`
- `python_binding_same_library=true`
- `native_backend_initialized=true`
- `cuda_support_direct=true` e `cuda_support_python=true`

**Non sono eseguiti download, compilazioni, pip, modifiche alla .venv,
caricamenti di modelli o inferenza.** La presenza del supporto CUDA e
l'inizializzazione del backend NON provano che un modello stia realmente
eseguendo layer sulla GPU: `gpu_offload_executed=false` e
`inference_executed=false` restano correttamente tali.

## Due metodi, una sola compilazione

- `ruth_runtime.cuda_adapter.CudaLlamaAdapter`: futura inferenza GGUF
  ad alto livello tramite il binding Python già installato.
- `ruth_runtime.direct_llama.DirectLlamaLibraryAdapter`: API C diretta
  in `ctypes`, con `open()` / `system_info()` / `close()` e gestione
  di `llama_backend_init/free`. Non implementa ancora caricamento GGUF,
  decodifica o generazione; tali funzioni rimangono un TO DO separato
  perché le strutture native di `llama.h` sono legate alla revisione ABI.

Entrambi sono vincolati alla stessa libreria fisica. Non viene prodotta
una seconda installazione nativa e non vengono creati due runtime divergenti.
In futuro si può distribuire una libreria standalone separata soltanto
verificando i suoi link dinamici e mantenendo lo stesso ABI e SHA256.

Il TO DO di `diskcache 5.6.3` / CVE-2025-69872 rimane aperto fino a
release upstream corretta, con `LlamaDiskCache` disabilitata.

## Verifiche successive

Dopo questo audit sarà necessario un modello GGUF reale autorizzato e
verificato per misurare offload, VRAM e inferenza. Il comando
`./ruth run` non sarà considerato operativo solo perché l'API C
si inizializza senza errori.
