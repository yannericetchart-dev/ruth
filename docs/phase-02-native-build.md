# Fase 02 — Build CUDA nativa automatica (wheel staged, senza installazione)

## Prerequisiti verificati sul PC

- 159 test superati prima di questo cambiamento; checkout esatto di `llama-cpp-python v0.3.36` / `vendor/llama.cpp` verificato.
- Toolkit CUDA 13.2, RTX 4050 `sm_89`, CMake 3.28, GCC 13.3, GNU Make 4.3.
- Otto wheel Python già scaricati e verificati con `./ruth setup --verify-candidate`.
- La CVE `diskcache` è un **TO DO non prioritario** fino a una release upstream corretta. La build non installa `diskcache`.

## Ricetta riproducibile e isolamento

I record `locks/python-cuda-3.12.build-inputs.json` fissano byte, nomi,
versioni, dimensioni e SHA-256 degli **otto** wheel risultanti dall'audit
già svolto su PyPI. Sono pin degli **input**, non approvazione dell'intero
runtime né attestazione che la vulnerabilità `diskcache` sia corretta.

Il comando unico per autorizzare la **sola compilazione locale** è:

```bash
./ruth setup --build-native
```

Dopo il preflight, Ruth:
1. Verifica Python 3.12 nella propria `.venv`, Toolkit/toolchain, spazio disco
   (almeno 8 GiB), commit Git esatti di entrambi i sorgenti e integrità dei wheel
   rispetto al manifest di pin versionato nel repository.
2. Crea una directory privata, con lock di concorrenza, in `.ruth-build/`.
3. Copia i sorgenti fissati in staging **senza alterare il checkout Git**. Blocca
   symlink non attesi e directory di output preesistenti.
4. Prepara in `.ruth-build/...` i tre wheel necessari alla compilazione
   (`scikit-build-core`, `packaging`, `pathspec`) usando
   `.venv/bin/python -m pip --isolated install --no-index --no-deps --target ...`.
   Questo **non installa** i pacchetti nella `.venv` e non installa
   `diskcache` o il binding.
5. Compila dal sorgente di staging:
   ```text
   .venv/bin/python -m pip --isolated wheel --no-index --no-deps --no-build-isolation ...
   CUDA_HOME=/usr/local/cuda-13.2
   CUDACXX=/usr/local/cuda-13.2/bin/nvcc
   CMAKE_GENERATOR=Unix Makefiles
   CMAKE_BUILD_PARALLEL_LEVEL=2
   CMAKE_ARGS=-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DLLAVA_BUILD=OFF
   ```
   Non usa rete, `apt`, la `.venv` globale o un wheel CPU precompilato.
6. Legge l'artefatto prodotto, richiede versione `0.3.36` e la presenza
   di **`libggml-cuda.so` e `libllama.so`** nel wheel; in caso contrario
   rifiuta la compilazione anziché ricorrere a una build CPU-only.
7. Registra SHA-256 e opzioni in
   `.ruth-build/native-cuda-0.3.36-sm89/build-result.json`, insieme al wheel;
   rimuove sorgenti copiati e strumenti temporanei dopo il successo.

**Successo atteso:** `status=native_cuda_wheel_built`,
`ggml_cuda_library_packaged=true`,
`installed_in_ruth_venv=false`,
`gpu_offload_executed=false`,
`inference_executed=false`.

Il controllo della presenza della libreria CUDA **non prova ancora**
che il binding possa eseguire kernel o inferenza sulla RTX 4050. Queste
prove sono una fase successiva. Un errore di build non autorizza l'utilizzo
del wheel CUDA precompilato né il fallback CPU. I file temporanei falliti
vengono eliminati; una build completa già presente non viene sovrascritta.

## Ulteriori passaggi per usare Ruth

Per installare il binding nella `.venv` serviranno una procedura
transazionale/di ripristino, l'approvazione dei pacchetti runtime e
un'autorizzazione d'installazione separata. La decisione di rimandare la
correzione `diskcache` **non** costituisce consenso implicito a installare
una dipendenza vulnerabile. Ruth mantiene `LlamaDiskCache` disabilitata.

Dopo l'installazione, il binding dovrà superare
`llama_supports_gpu_offload()` e un'inferenza con modello GGUF reale,
comprovando uso GPU. Fino a quel momento `./ruth run` resta bloccato.
