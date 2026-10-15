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

## Exact staged allowlist

Only the pinned Gemma 3 BF16 weight file, mode=context, a single CUDA
device, `n_ctx=256`, `n_batch=n_ubatch=64`, `swa_full=on`,
`cuda_preinit=false`, one sequence and KQV offload enabled.
Two experimental combinations:

1. **F16/F16, Flash OFF** — baseline context allocation, no inference.
2. **BF16/BF16, Flash ON** — experimental BF16 context allocation, no inference.

No mixed precision, F32, quantized KV, Flash AUTO, other batch, longer
contexts or automatic fallback. The shared gate runs both in
`probe_model_load` (before environment checks) and in the isolated
`load_only_worker` (before loading llama_cpp). Successful completion
requires full model GPU offload; a CPU or partial GPU fallback is never
recorded as a successful load. Each explicit `--allow-gpu-load` command
is permission for **allocation only**, not tokenization or inference.

## First verify all offline tests — use .venv/bin/python

```bash
cd ~/Dev/ruth-model-profiling
git fetch origin
git switch feat/gemma3-native-context-probes
git pull --ff-only origin feat/gemma3-native-context-probes
.venv/bin/python -m unittest \
  tests.test_phase11_load_only_probe \
  tests.test_phase22_gemma3_bf16_onboarding \
  tests.test_phase24_gemma3_kv \
  tests.test_phase26_gemma3_context_probe -v
```

All tests should complete before any GPU command. No real CUDA is invoked
by the new unit tests.

## Two explicit isolated CUDA allocations

Start with the F16 baseline, review its JSON/log-derived outcome, and
only then consider the BF16 experiment.

```bash
.venv/bin/python -m ruth_runtime models probe-load \
  --purpose production --family gemma --model gemma-3 \
  --version gemma-3-1b-it --format gguf --quantization BF16 \
  --load-mode context --n-ctx 256 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn off --type-k F16 --type-v F16 \
  --allow-gpu-load --no-git-sync
```

Only after assessing the first result:

```bash
.venv/bin/python -m ruth_runtime models probe-load \
  --purpose production --family gemma --model gemma-3 \
  --version gemma-3-1b-it --format gguf --quantization BF16 \
  --load-mode context --n-ctx 256 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn on --type-k BF16 --type-v BF16 \
  --allow-gpu-load --no-git-sync
```

Both use a child process, native ABI verification and repeat GPU
preflight, no download, no CUDA inference and no production promotion.
`--no-git-sync` retains the raw evidence locally for inspection before
owner-publishing any sanitized information.

## Evidence and promotion rules

The parent collects native logs, KV/compute buffer allocation records,
actual context initialization, offload layers, GPU memory observations and
failure categorization. The Gemma 3 recorder normalizes its *own*
GGUF-derived layer geometry rather than reusing Gemma 2's KV parameters.
The public proposal validator re-computes the same source geometry from
the pinned canonical profile before allowing Git sync.

A successful context allocation is **not** proof of runtime Flash kernel
selection, inference correctness, numerical precision, performance, or
production eligibility. The `kv_attention_policy` and the default
production candidate are unchanged. If CUDA OOM/unsupported native type
occurs, stop and inspect; do not silently switch context, GPU, model,
dtype or Flash settings.
