# Gemma 3 1B: pinned CUDA attention kernels, build audit, KV precision policy

## Scope and provenance

This is **read-only source-derived planning**, not native context approval.

- Ruth native llama-cpp-python: `0.3.36`, source commit `1652066e0af45f2313b339670ef9555e8a54e545`.
- llama.cpp submodule: `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- CUDA toolkit declared by the Ruth builder: `/usr/local/cuda-13.2`; compute architecture `89`.
- Model GGUF: pinned Gemma 3 1B IT BF16 SHA256 `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`.

## Source observations at the pinned llama.cpp commit

1. `ggml/CMakeLists.txt`: `GGML_CUDA_FA` defaults ON. `GGML_CUDA_FA_QUANTS` defaults to
   `q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16`. This is the **source default**, not a
   verified compilation result for an installed wheel.
2. `ggml/cmake/common.cmake`: sets explicit `GGML_CUDA_FA_<K>_<V>` template macros;
   unlisted pair combinations do not automatically get vector specializations.
   `f16-f16` is added even if omitted from a nonempty custom list.
3. `ggml/src/ggml-cuda/fattn.cu`: runtime dispatcher chooses vector (VEC),
   tensor-core MMA_F16, tile (TILE), or no Flash kernel. No universal
   bf16 -> f16 fallback statement is correct.
4. `fattn.cu`: vector path missing a compiled K/V instance logs a warning and
   routes through the f16 vector specialization, converting as needed. F32
   is mapped to a compatible f16 vector specialization. MMA_F16 and TILE
   request temporary f16 K/V buffers independent of original KV precision.
5. `src/llama-context.cpp`: quantized **V** requires Flash Attention ON;
   non-quantized F16/BF16/F32 values do not trigger this restriction.
6. In this pinned tree, `fattn-wmma-f16.cuh` is not a separate file.
   Do not infer kernel availability from unrelated upstream versions.

Native context success, numerical fidelity, performance and whether one
specific vector specialization was compiled remain unverified.

## Ruth's current native builder

`ruth_runtime/native_builder.py` specifies:

```text
-DGGML_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=89
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc
-DLLAVA_BUILD=OFF
```

Neither `GGML_CUDA_FA` nor `GGML_CUDA_FA_QUANTS` is explicitly overridden.
At the pinned upstream defaults, **both** f16-f16 and bf16-bf16
vector templates are expected in an unmodified build. However, the
manifest's CMAKE_ARGS is not proof of effective CMakeCache options,
installed .so contents, CUDA dispatch or numerical correctness.

The wheel lock contains SHA256 and commit identity, but not a separately
verified record of compiled FlashAttention template combinations.

## Read-only local audit (run inside project root)

Always use Ruth's project Python:

```bash
cd ~/Dev/ruth-model-profiling
.venv/bin/python -m unittest tests.test_phase24_gemma3_kv -v
.venv/bin/python -m ruth_runtime models kv-plan \
  --purpose production --family gemma --model gemma-3 \
  --version gemma-3-1b-it --quantization BF16 \
  --n-ctx 256 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn on --type-k BF16 --type-v BF16

.venv/bin/python - <<'PY'
from pathlib import Path
import json
from ruth_runtime.native_builder import CUDA_ARGUMENTS
print("Pinned Ruth CMAKE_ARGS:", CUDA_ARGUMENTS)
manifest = Path(".ruth-build/native-cuda-0.3.36-sm89/build-result.json")
if manifest.is_file():
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    for key in ("llama_cpp_commit", "cuda_architecture", "cmake_args",
                "wheel_sha256"):
        print(f"{key}: {obj.get(key)}")
else:
    print("build-result.json: absent; no local artifact proof")
cache_files = list(Path(".ruth-build").glob("**/CMakeCache.txt"))
if cache_files:
    for cache in cache_files:
        print("CMakeCache:", cache)
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith(("GGML_CUDA_FA:", "GGML_CUDA_FA_QUANTS:",
                                "CMAKE_CUDA_ARCHITECTURES:")):
                print(line)
else:
    print("CMakeCache.txt: absent; actual compiled FA options UNKNOWN")
PY
```

The builder deliberately removes intermediate build sources/artifacts
after success. Consequently, CMakeCache may be missing. A missing
CMakeCache must remain `unknown`, never `bf16 kernel verified`.

## Read-only precision comparison policy

| Flash mode | KV pair | Interpretation | Native permission |
| --- | --- | --- | --- |
| OFF | F16/F16 | Existing reference baseline | Unchanged, not newly approved |
| OFF | BF16/BF16 | Experimental non-Flash storage | Not approved |
| OFF | F32 or mixed | Diagnostic logical-memory estimate | Not approved |
| ON | F16/F16 | Reference Flash comparison candidate | Not approved |
| ON | BF16/BF16 | Source-default vector specialization candidate | Not approved |
| ON | F32 or mixed | Conversion/fallback-risk diagnostic | Not approved |

For Flash ON, both the prefill and decode phases can select different
kernels. The source-default vector pair does **not** guarantee direct
BF16 execution: MMA_F16/TILE may convert to F16. Flash ON may require
additional temporary buffers that are **not** part of logical KV bytes.
Flash OFF does not use the Flash kernel dispatch.

Ruth's `kv-plan` accepts explicit Flash ON/OFF to produce a
`kv_attention_policy` in addition to storage estimates. `auto`
remains blocked for Gemma3 planning until effective runtime mode can
be observed. The `probe-load` Gemma3 context gate is unchanged and
no production dtype switch is authorized.

### Required evidence before any native comparison

- Verify exact installed wheel identity and build inputs.
- Recover effective CMake flags or rebuild in a controlled, separately
  approved process with explicit `GGML_CUDA_FA` and `GGML_CUDA_FA_QUANTS`.
- Obtain native context allocation evidence with no token generation;
  verify GPU fit and kernel support per dtype and Flash mode.
- Then obtain separate, explicitly authorized prefill/decode timing,
  dispatch and numeric-fidelity evidence.
