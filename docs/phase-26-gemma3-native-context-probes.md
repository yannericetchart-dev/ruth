# Phase 26 — Gemma 3 1B BF16: staged native context allocation

## Verified prerequisites and limits

- Pinned GGUF `gemma-3-1b-it/gguf/BF16` SHA256:
  `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`.
- Pinned llama.cpp commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f`
  through `llama-cpp-python 0.3.36`; local native ABI, hashes and GPU
  readiness are independently checked immediately before launching a worker.
- The user's read-only binary audit established wheel SHA256
  `ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`;
  `libggml-cuda.so` SHA256
  `7851341a4e83575a3fa5039e1d87e25022cea82cd07948aa1d2bf85d1b3f5ca1`,
  and sm_89 cubin entries for `flash_attn_ext_vec<256, ..., BF16, BF16, ...>`.
  These are host-reported historical observations, **not evidence of a
  native context allocation or kernel actually dispatched**.
- The Python GGML enum at pinned llama.cpp uses F16=1, BF16=30. The
  pinned Python binding omits a named `GGML_TYPE_BF16`, so Ruth uses
  pinned numeric value 30 (rejecting contradictory definitions) only
  after its native/ABI gate.

## Exact 18-way allocation-only matrix

Only the pinned Gemma 3 BF16 weight file, mode=context, a single CUDA
device, `n_ctx=256`, `n_batch=n_ubatch=64`, `swa_full=on`,
`cuda_preinit=false`, `n_seq_max=1`, `kv_unified=false`,
GPU KQV offload enabled. Independent Key and Value storage types each
belong to `F16`, `BF16`, `F32`. Flash must be explicitly ON or OFF.

**3 Key × 3 Value × 2 Flash = 18 requested configurations.**

The two original probes have already been observed by the user:

- #1 F16/F16, Flash OFF: context initialized, 27/27 GPU layers, 6.5 MiB
  native KV, no inference.
- #2 BF16/BF16, Flash ON: context initialized, 27/27 GPU layers, 6.5 MiB
  native KV, no inference.

All other configurations are **untested**, not disallowed by a measured
incompatibility. With Flash ON, an initialized context does not prove a
VEC BF16 kernel executed. During prefill or decode, an MMA/TILE kernel
could use a temporary F16 conversion. F32 and mixed KV combinations
are therefore diagnostics, not production recommendations.

| Case | Flash | K | V | KV logical MiB |
|---:|:---:|:---:|:---:|---:|
| 1 | OFF | F16 | F16 | 6.50 |
| 2 | ON | BF16 | BF16 | 6.50 |
| 3 | OFF | BF16 | BF16 | 6.50 |
| 4 | OFF | BF16 | F16 | 6.50 |
| 5 | OFF | BF16 | F32 | 9.75 |
| 6 | OFF | F16 | BF16 | 6.50 |
| 7 | OFF | F16 | F32 | 9.75 |
| 8 | OFF | F32 | BF16 | 9.75 |
| 9 | OFF | F32 | F16 | 9.75 |
| 10 | OFF | F32 | F32 | 13.00 |
| 11 | ON | BF16 | F16 | 6.50 |
| 12 | ON | BF16 | F32 | 9.75 |
| 13 | ON | F16 | BF16 | 6.50 |
| 14 | ON | F16 | F16 | 6.50 |
| 15 | ON | F16 | F32 | 9.75 |
| 16 | ON | F32 | BF16 | 9.75 |
| 17 | ON | F32 | F16 | 9.75 |
| 18 | ON | F32 | F32 | 13.00 |

The gate runs both in `probe_model_load` (before environment checks)
and in `load_only_worker` (before loading llama_cpp). Unknown types,
Flash AUTO, quantized KV, other batch sizes and contexts stay blocked.
The Gemma 2 legacy Value-type restriction remains intact.

## Check offline tests first — always use .venv/bin/python

```bash
cd ~/Dev/ruth-model-profiling
git fetch origin
git switch feat/gemma3-kv-matrix-18
git pull --ff-only origin feat/gemma3-kv-matrix-18

.venv/bin/python -m unittest \
  tests.test_phase11_load_only_probe \
  tests.test_phase14_profile_registry \
  tests.test_phase15_profile_contribution \
  tests.test_phase22_gemma3_bf16_onboarding \
  tests.test_phase24_gemma3_kv \
  tests.test_phase26_gemma3_context_probe \
  tests.test_phase26_gemma3_context_matrix -v
```

All new tests use mocks and temporary fixtures: no real CUDA allocations,
GGUF loading, tokenization, or inference.

## Read-only matrix plan (safe default)

```bash
.venv/bin/python -m ruth_runtime.gemma3_context_matrix
```

Prints the exact 18-case order, types, derived logical KV costs and
identities; **does not invoke any worker**. Run this first.

## Explicit one-case GPU probe (recommended initially)

```bash
.venv/bin/python -m ruth_runtime.gemma3_context_matrix \
  --case 3 --execute --allow-gpu-load
```

This runs **only BF16/BF16 with Flash OFF**, the first of the 16 new
configurations, and preserves the local evidence. Examine the JSON
for `native_context_verified`, expected key/value types in both
global and SWA logs, real CUDA buffer allocations and any native
failures. This requires an explicit, real GPU allocation authorization.

To inspect the next case separately, substitute `--case 4`.

## Explicit sequential remainder (only after reviewing case 3)

```bash
.venv/bin/python -m ruth_runtime.gemma3_context_matrix \
  --start-at 4 --execute --allow-gpu-load
```

This requests cases 4–18 sequentially. Each case invokes Ruth's
`probe-load` as a separate isolated Python subprocess, which rechecks
pinned GGUF integrity, wheel/ABI identity, the exact GPU and free
capacity. It **stops at the first error, timeout, partial CPU/GPU
offload, mismatched native cache dtype, invalid context geometry,
failure to archive local evidence, or other unverified result**.
No Ruth-level cache precision retry, CPU fallback, parallel CUDA work,
model download, Git publication or inference occurs. Internal native
attention-kernel conversion (e.g., temporary BF16/F32-to-F16) is NOT
excluded by a context-only test and remains for later dispatch profiling.

Use `--start-at N` to resume a stopped batch. Skipped cases are not
claimed to have passed in the new run. To repeat the entire comparable
matrix later, explicitly use `--start-at 1`.

Every worker calls `probe-load --allow-gpu-load --no-git-sync`;
the top-level `--execute --allow-gpu-load` opts into multiple separate
GPU allocations. Never run the executable mode unattended if you need
to monitor native allocation errors or temperature/host contention.

## Evidence and promotion rules

The parent collects native logs, KV/compute buffer allocation records,
actual context initialization, offload layers, GPU memory observations and
failure categorization. The Gemma 3 recorder normalizes its *own*
GGUF-derived layer geometry rather than reusing Gemma 2's KV parameters.
The public proposal validator re-computes the same source geometry from
the pinned canonical profile before allowing Git sync. Gemma3 context
measurements also carry the SHA256 of the exact pinned native CUDA wheel;
Git validation rejects a missing or mismatched wheel hash, and canonical
context observation identity includes that hash so different native builds
cannot silently share measurements. Historical Gemma2 context records retain
their existing identity and validation schema.

A successful context allocation is **not** proof of runtime Flash kernel
selection, inference correctness, numerical precision, performance, or
production eligibility. The `kv_attention_policy` and the default
production candidate are unchanged. If CUDA OOM/unsupported native type
occurs, stop and inspect; do not silently switch context, GPU, model,
dtype or Flash settings.
