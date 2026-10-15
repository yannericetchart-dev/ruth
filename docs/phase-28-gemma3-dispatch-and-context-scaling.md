# Phase 28 — Gemma 3 CUDA dispatch evidence and long-context scaling

**Experimental only.** This branch does not amend the Phase 26 allocation
gate, the Phase 27 short-token benchmark, the actual production runtime or
the canonical profile. The user has successfully performed Phase 27 native
inference on all 18 Flash/K/V combinations. CUDA kernel dispatch has not
yet been measured.

## Important measurement correction

The initial Phase 28 F16/F16 Flash OFF, `n_ctx=256` run completed native
inference, with 128 prefill tokens, 16 decode calls, full offload and
verified KV geometry. However, the original worker requested logits only
on the final decode call, and its reported **8.6146 ms/token / 116.08
tokens/s** was consequently biased low relative to actual autoregressive
decode. This is not evidence for a 116 tok/s production setting.

Starting with the corrected worker, the output projection is requested
on every decode step, and the parent **rejects** results lacking this
explicit invariant. All Phase 28 performance measurements used for
comparisons must come from the corrected commit or later and use the
same full-logits-on-every-step methodology. The old result remains valid
as a limited native inference/VRAM smoke test.

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
timing; then 16 identical, teacher-forced decode inputs follow. **Every
decode call explicitly requests full-vocabulary logits output**, even
though only the final row is saved. This includes the output projection
cost that an autoregressive model ordinarily needs at every token.
The worker reports `decode_logits_requested_per_token=true` and
`decode_logits_requested_count=16`, verified again by the parent.

Latency includes host-side native-call/synchronization overhead. The
reported `prefill_ms` is the **sum of all chunks** (different workload
from the Phase 27 seven-token prefill); `decode_ms_per_token_median`
summarizes the per-call distribution. `decode_total_ms_median` is the
median of the three complete 16-token rounds, and
`decode_tokens_per_second` is computed as 16 divided by that round
duration, not the inverse per-call median. These metrics are not a
natural sampling benchmark because the decode tokens are teacher-forced.
Samples of GPU process memory are not a guaranteed instantaneous peak.

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

## First actual Nsight dispatch evidence — BF16/BF16 Flash ON, n_ctx=2048

Nsight Systems 2025.6.3 successfully captured and exported the user's
experiment to `.ruth-build/phase28/bf16-on-ctx2048.sqlite`. The original
offline reader observed **24,120 CUDA kernel launch records**. Its
`FLASH_OTHER` labels were incomplete: the pinned llama.cpp source shows
that `flash_attn_ext_f16` is launched by the MMA_F16 branch. The
post-capture reader update fixes the classification without modifying the
original SQLite artifact or rerunning inference.

| Observed CUDA symbol | Family | Calls | Sum of GPU kernel duration |
|---|---|---:|---:|
| `flash_attn_ext_f16<256,256,16,4,...>` | FLASH_MMA_F16 | 728 | 19.5141 ms |
| `flash_attn_ext_vec<256,1,30,30,...>` | FLASH_VEC | 104 | 1.4721 ms |
| `flash_attn_stream_k_fixup_uniform<256,16,4>` | FLASH_FIXUP | 728 | 4.7190 ms |
| `flash_attn_combine_results<256>` | FLASH_COMBINE | 104 | 0.2247 ms |

The ggml enum in the pinned `ggml/include/ggml.h` assigns
`GGML_TYPE_BF16 = 30`: the vector kernel explicitly uses BF16 K and V.
The pinned CUDA dispatcher selects `BEST_FATTN_KERNEL_MMA_F16` separately
from `BEST_FATTN_KERNEL_VEC`, and the MMA_F16 path requires F16 K and V
operands. This accounts for both a BF16-native vector attention kernel
and a separate F16-specialized attention kernel within a BF16-cache run.
The captured `convert_unary_cont_vec4<__nv_bfloat16,__half>`
appeared **1,456** times, consistent with BF16-to-F16 conversion, but
count correlations alone cannot establish which attention launch
consumed each converted tensor.

**Do not infer** prefill/decode mapping from the kernel launch
frequencies. The report has no verified NVTX/CUPTI phase correlation.
The times above are sums of the selected CUDA kernels across the
profiled worker process (including warmup/repeats), **not** inference
latency totals, nor speedups over another candidate. The production
profile remains unchanged.

After pulling the updated reader, re-analyze **offline**:

```bash
.venv/bin/python -m unittest tests.test_phase28_gemma3_context_dispatch -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_dispatch \\
  --sqlite .ruth-build/phase28/bf16-on-ctx2048.sqlite
```

The updated report contains `main_attention_families_observed` and
`attention_family_summary`, separating true attention kernels from
auxiliary fixup/combine kernels. No new CUDA run is needed.

## Promotion gate

Phase 28 produces *diagnostic* context, timing, memory and CUDA-kernel
evidence, never a production policy selection. Longer-context numerical
accuracy, repeated workload timing, per-phase kernel attribution and
headroom remain open until their native evidence is inspected.
