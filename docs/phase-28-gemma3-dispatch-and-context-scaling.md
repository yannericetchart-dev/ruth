# Phase 28 — Gemma 3 CUDA dispatch evidence and long-context scaling

**Experimental only.** This branch does not amend the Phase 26 allocation
gate, the Phase 27 short-token benchmark, the actual production runtime or
the canonical profile. The user has successfully performed Phase 27 native
inference on all 18 Flash/K/V combinations. CUDA kernel dispatch has not
yet been measured.

## Exact experiment boundaries

Model: pinned Gemma 3 1B IT BF16 GGUF, previously verified SHA256,
and pinned llama.cpp `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
Only NVIDIA GPU with verified CUDA wheel/dual ABI, one sequence, 27/27
full offload, SWA FULL and `n_batch=n_ubatch=64`.

Phase 28 deliberately restricts the new, longer contexts to:

| n_ctx | Fixed prefill tokens | Teacher-forced decode | KV F16/BF16 MiB | KV F32/F32 MiB |
|---:|---:|---:|---:|---:|
| 256 | 128 | 16 | 6.5 | 13 |
| 512 | 384 | 16 | 13 | 26 |
| 1024 | 768 | 16 | 26 | 52 |
| 2048 | 1536 | 16 | 52 | 104 |

These are **logical** KV sizes, not total or peak VRAM usage.
The four *independently specified* cache/Flash candidates are:

- `f16-off`: F16/F16, Flash OFF (historical conservative control)
- `bf16-on`: BF16/BF16, Flash ON (historical promising candidate)
- `f32-off`: F32/F32, Flash OFF (higher precision *storage*, not ground truth)
- `mixed-on`: BF16/F16, Flash ON (distinct mixed-type control)

There are 16 permitted candidate×context pairs, and **no automatic
progression** between them: one explicit selection per invocation.
Other dtype pairs are not concluded incompatible: the earlier 18-way
matrix remains the reference for their short-token compatibility.

### Methodology and measurement

One warm-up followed by three timed repetitions, each with memory cleared.
The public, deterministic prompt uses the same immutable tokenizer and a
repeated fixed token sequence (not user content). A prefill is split into
ordered 64-token chunks and every `llama_decode` is synchronized before
timing; then 16 identical, teacher-forced decode inputs follow.

Latency includes host-side native-call/synchronization overhead. The
reported `prefill_ms` is the **sum of all chunks** (different workload
from the Phase 27 seven-token prefill); `decode_ms_per_token` is the
medians over repeated decode calls. Samples of GPU process memory are not
a guaranteed instantaneous peak.

Only the last-prefill and final-decode *full-vocabulary* F32 logits rows
are saved to local compressed `.npz` files. The worker retains the token
IDs, prompt hash and local SHA256 for offline comparison. It does not
sample free-form text. Native inference **does** execute.

The parent verifies pinned GGUF integrity, wheel provenance/dual ABI,
NVIDIA UUID/exclusivity and usable VRAM, then checks GPU telemetry **again**
before launching the worker. It requires **at least 3072 MiB free** at
both preflights, a conservative floor, not a proof the extended context
fits. A timeout, native failure, partial offload, wrong KV types, or
unexpected native context geometry **stops without fallback**.

## 1. Checkout and offline tests (no CUDA)

```bash
cd ~/Dev/ruth-model-profiling
git fetch origin
git switch feat/gemma3-phase28-context-dispatch
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase26_gemma3_context_probe \
  tests.test_phase26_gemma3_context_matrix \
  tests.test_phase27_gemma3_benchmark \
  tests.test_phase28_gemma3_context_dispatch -v
```

## 2. Read-only experiment plan

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28
```

This enumerates the 16 permitted pairs and theoretical KV allocation;
it performs no model load, CUDA execution or network access. Filter one
pair with `--candidate f16-off --n-ctx 256`.

## 3. Explicit first long-context GPU test — 128-token control

**The following runs real CUDA model inference.** Opt-in is required.

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate f16-off --n-ctx 256 \
  --execute --allow-gpu-inference
```

A verified response is `native_long_context_inference_observed`
with measured native `n_ctx=256`, expected SWA/global KV dtype/layers,
`offloaded_layers=27`, 128 prefill tokens, 16 decode tokens, latency
summaries, GPU memory sampled, and a hashed local numerical artifact.

Only after reviewing this result, perform the isolated BF16 Flash
comparison under the identical workload:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate bf16-on --n-ctx 256 \
  --execute --allow-gpu-inference
```

If both complete, advance to `--n-ctx 512` **individually**, then 1024
and 2048; the gate never silently increases context or falls back to
CPU. Repeat `f32-off` and `mixed-on` only after the baselines succeed.

## 4. Offline long-context numerical comparison

Use two `logits_artifact.relative_path` values produced at the **same
context length**, for example:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_compare \
  --baseline '.ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-off_k-F16_v-F16_EXAMPLE.npz' \
  --candidate '.ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-on_k-BF16_v-BF16_EXAMPLE.npz'
```

Replace EXAMPLE with paths reported by the real native worker. The
comparator refuses different context lengths or prompt/decode token
sequences, and computes full-vocabulary KL, JS, RMSE, top-1 and maximum
logit difference *offline*. Its two rows are **last prefill** and
**final decode**, not every intermediate token as in Phase 27.

## 5. Capture and inspect actual CUDA dispatch — separate from timings

The pinned llama.cpp source contains FlashAttention VEC, MMA_F16 and TILE
paths. Its Flash graph also **casts F32 K/V operands to F16** before
`ggml_flash_attn_ext`. Neither compiled specialization nor a
successful Flash ON execution proves which GPU kernel ran.

First verify that Nsight Systems is installed; nothing is installed
automatically:

```bash
nsys --version
nsys profile --help
nsys export --help
```

Only after a successful ordinary benchmark, optionally capture one
*explicitly authorized* GPU run. It may be significantly slower:

```bash
mkdir -p .ruth-build/phase28
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o .ruth-build/phase28/bf16-on-ctx256 \
  .venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate bf16-on --n-ctx 256 \
  --execute --allow-gpu-inference
```

Confirm the produced report file and check the installed CLI's export
syntax (`nsys export --help`). A typical invocation is:

```bash
nsys export --type sqlite \
  --output .ruth-build/phase28/bf16-on-ctx256.sqlite \
  .ruth-build/phase28/bf16-on-ctx256.nsys-rep
```

Then inspect observed kernel names without CUDA execution:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_dispatch \
  --sqlite .ruth-build/phase28/bf16-on-ctx256.sqlite
```

The inspector checks native CUPTI kernel records and groups exact
captured names into VEC, MMA_F16, TILE or unclassified kernels.
**It does not correlate device kernels with prefill versus decode.**
NVTX markers are optional and were unavailable in the previous local
test. Even with kernel names in a trace,
`prefill_decode_attribution_verified=false`. A separate verified
CUPTI/NVTX timing join is necessary for that stronger claim.

Use unprofiled executions for latency comparisons; profiling changes
runtime overhead. No kernel-name observation alone approves production.

## Promotion gate

Phase 28 produces *diagnostic* context, timing, memory and CUDA-kernel
evidence, never a production policy selection. Longer-context numerical
accuracy, repeated workload timing, per-phase kernel attribution and
headroom remain open until their native evidence is inspected.
