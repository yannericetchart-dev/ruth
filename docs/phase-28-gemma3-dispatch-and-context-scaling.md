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

## Second actual Nsight dispatch evidence — F16/F16 Flash OFF, n_ctx=2048

The same user successfully profiled and exported the F16/F16 Flash OFF
workload to `.ruth-build/phase28/f16-off-ctx2048.sqlite`.
The worker again verified 27/27 CUDA offload, 1536 prefill tokens, 16
logits-enabled decode tokens, and 52 MiB logical F16 KV. The saved logits
SHA256 `e0f1c3f29fca765992c0d4f93d345cbabaf1b97d6e97cb5c0bd3268f2b46e3c4`
matches the unprofiled F16 run.

Nsight Systems 2025.6.3 generated a profile with 141,195 trace events
(the Flash ON trace had 123,039 events; **these are all trace events,
not counts of CUDA kernel launches**, so they must not be used to
quantify per-token GPU overhead). The native `cuda_gpu_kern_sum`
report records these source-consistent *non-Flash operator* symbols:

| Observed kernel symbol | Calls | Sum of GPU kernel duration |
|---|---:|---:|
| `cutlass::Kernel2<cutlass_80_tensorop_s1688gemm...>` | 728 | 14.9653 ms |
| `soft_max_f32<true,0,0,float>` | 520 | 13.3291 ms |
| `soft_max_f32<true,1024,1024,float>` | 104 | 2.8050 ms |
| `soft_max_f32<true,512,512,float>` | 104 | 1.0559 ms |
| `soft_max_f32<true,256,256,float>` | 104 | 0.6658 ms |

The pinned source `src/llama-graph.cpp` (approximately lines 2688–2732)
explicitly builds a K@Q matmul, `ggml_soft_max_ext` and a V@attention
matmul when Flash is disabled. The pinned CUDA softmax implementation
contains the observed `soft_max_f32` kernel. However, the CUTLASS GEMM
symbols are **generic** and may represent other matrix multiplications,
not just attention. The CUDA trace still lacks a graph-to-kernel join
and prefill-vs-decode phase attribution.

The post-trace offline reader accepts a newly explicit
`--expected-flash off` mode only when it sees recognizable softmax
and CUTLASS operators **and** no FlashAttention family. It also accepts
`--expected-flash on` only if it sees a main FlashAttention kernel;
an incorrect mode label causes a nonzero exit. This is a limited
source-informed consistency check, **not** native proof of each
attention-graph operation.

To re-read both existing traces, without any GPU work:

```bash
.venv/bin/python -m unittest tests.test_phase28_gemma3_context_dispatch -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_dispatch \\
  --sqlite .ruth-build/phase28/f16-off-ctx2048.sqlite \\
  --expected-flash off

.venv/bin/python -m ruth_runtime.gemma3_phase28_dispatch \\
  --sqlite .ruth-build/phase28/bf16-on-ctx2048.sqlite \\
  --expected-flash on
```

`native_kernel_names_verified` in this version remains backward-compatible
and specifically denotes **main FlashAttention kernel names**, so it is
false for a healthy Flash OFF run. For Flash OFF, inspect
`expected_flash_consistent_with_kernel_names`, `operator_kernel_summary`,
and the explicitly false `nonflash_graph_kernel_attribution_verified`.
Do not promote the runtime based on this evidence.

## NVTX3 opt-in bridge and fail-closed phase attribution (new)

The user's host has no legacy `nvToolsExt` shared library or Python
`nvtx` package, but **does** have the CUDA NVTX3 header at
`/usr/local/cuda/include/nvtx3/nvToolsExt.h`. NVTX3 is header-only:
there is no need to install new Python packages, change llama.cpp,
download code, or modify production runtime. NVIDIA documents
exporting the C API with `#define NVTX_EXPORT_API` for ctypes consumers.

After pulling the Phase28 branch, run offline unit tests:

```bash
.venv/bin/python -m unittest tests.test_phase28_gemma3_context_dispatch -v
```

**Manual, local compilation** (no inference, no automatic compilation
inside the worker; output deliberately stays in ignored `.ruth-build`):

```bash
mkdir -p .ruth-build/phase28
cc -std=c11 -O2 -fPIC -shared -I/usr/local/cuda/include \
  -o .ruth-build/phase28/libruth_phase28_nvtx3.so \
  ruth_runtime/native/phase28_nvtx3_export.c -ldl

nm -D .ruth-build/phase28/libruth_phase28_nvtx3.so | \
  grep -E 'nvtxRange(PushA|Pop)'
```

**Explicitly instrumented profiling**, while retaining both Phase28
GPU-inference authorization flags:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  -o .ruth-build/phase28/bf16-on-ctx2048-nvtx3 \
  env RUTH_PHASE28_NVTX3_BRIDGE=1 \
  .venv/bin/python -m ruth_runtime.gemma3_phase28 \
    --candidate bf16-on --n-ctx 2048 \
    --execute --allow-gpu-inference

nsys export --type sqlite \
  --output .ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite \
  .ruth-build/phase28/bf16-on-ctx2048-nvtx3.nsys-rep
```

The worker now attempts to load exactly
`.ruth-build/phase28/libruth_phase28_nvtx3.so` **only** when
`RUTH_PHASE28_NVTX3_BRIDGE=1` is present. A missing/invalid library
then causes an error rather than silently treating the run as marked.
The option does not affect normal Phase28 runs, Phase27, or production.
Even a successful `nvtx_ranges_available=true` only means the worker
loaded the NVTX C API; **it is not proof the profiler captured ranges**.

Finally correlate the new **local SQLite** export, without CUDA:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_attribution \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite \
  --candidate bf16-on --n-ctx 2048
```

The attribution reader requires NVTX range events with eventType 59,
the expected Phase28 labels (`RUTH_GEMMA3_PHASE28_PREFILL` and
`..._DECODE`), and exact range counts: 96 prefill chunk ranges and 64
decode ranges at 2048 context, including warmup. It links each
recognized attention kernel through its CUPTI `correlationId` to a
CUDA runtime API call whose **entire CPU interval** lies inside exactly
one same-thread NVTX range. It does not assume CUDA device start/end
timestamps lie inside a CPU range. Unmapped/ambiguous kernel records
cause `prefill_decode_attribution_verified=false` and exit code 2.

NVIDIA documents this approach in the Nsight Systems guide (example
"Rename CUDA Kernels with NVTX" linking `NVTX_EVENTS.globalTid`,
`CUPTI_ACTIVITY_KIND_RUNTIME.globalTid`, and GPU kernel
`correlationId`). SQLite schema version changes or GPU Graph launch
correlation may cause a legitimate trace to remain unverified; do
**not** change verification thresholds or fabricate a success flag to
accommodate them. Inspect the actual export first. The report never
qualifies a production profile.

## First native NVTX/CUPTI phase attribution — success at n_ctx=2048

The user's explicit NVTX3 bridge compilation succeeded and exported
`nvtxRangePushA` and `nvtxRangePop`. Profiling BF16/BF16 Flash ON
at `n_ctx=2048` reported `nvtx_ranges_available=true`, genuine
27/27 GPU offload, and logits snapshot SHA256
`5629cf4d0e7437e12af1ec9412309e60e76192d85c5e7a822dcd4c7695e613d0`,
identical to the unprofiled BF16 baseline.

Exported as `.ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite`,
the Phase28 attribution reader successfully joined CUPTI kernel
records to CUDA runtime calls within the intended NVTX CPU ranges:

| NVTX phase | Observed kernel | Calls | GPU time (sum) |
|---|---|---:|---:|
| Prefill | FLASH_MMA_F16 | 728 | 19.6109 ms |
| Prefill | FLASH_FIXUP | 728 | 4.7551 ms |
| Decode | FLASH_VEC | 104 | 1.4181 ms |
| Decode | FLASH_COMBINE | 104 | 0.2127 ms |

There were **96 prefill** and **64 decode** NVTX ranges (four
warmup+repeat trials), `attributed_attention_kernel_records=1664`,
`unattributed_attention_kernel_records=0`,
`ambiguous_cuda_correlations=0`, and
`prefill_decode_attribution_verified=true`.

**Important scope limitation:** this success verifies assignment of
all *recognized CUPTI kernel records* to the NVTX phases; it does
**not** prove that every attention operation executed throughout all
96 prefill chunks and 64 decode passes appears as an individually
correlatable runtime kernel record. In fact, the main-kernel counts
are 728 prefill = 7 × 26 layers × 4 trials and 104 decode =
1 × 26 layers × 4 trials. The pinned llama.cpp source defaults
`GGML_CUDA_GRAPHS` to ON and can use `cudaGraphLaunch`
(`ggml/src/ggml-cuda/ggml-cuda.cu`). Graph replay is a plausible
explanation, **not yet directly verified in the actual trace**.

Accordingly, the reader now exposes
`attribution_scope="recognized_cupti_kernel_launch_records"`,
`complete_attention_workload_coverage_verified=false`, and
`cuda_graph_replay_assessed=false` even when the narrower phase
attribution returns true. Never interpret this report as exhaustive
per-layer inference accounting, per-phase end-to-end latency, or
production qualification.

Next inspect CUPTI CUDA runtime graph launch APIs and any graph
activity tables in the *existing SQLite export* before attempting
an exact all-decode-call coverage proof. A separate explicitly
instrumented F16/F16 Flash OFF trace would complete the direct
phase-by-phase comparison without relying on kernel name counts.

## CUDA Graph executions confirmed in NVTX3 capture; replay audit

The user inspected the already-exported
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite`
and found:

| Runtime graph-related API | Calls |
|---|---:|
| `cudaGraphLaunch_v10000` | 128 |
| `cudaGraphExecUpdate_v10020` | 28 |
| `cudaGraphDestroy_v10000` | 28 |
| `cudaGraphInstantiate_v12000` | 2 |
| `cudaGraphExecDestroy_v10000` | 2 |

Both `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` and `CUDA_GRAPH_EVENTS`
exist in the export. **CUDA Graph replay is now directly observed**,
not merely hypothesized from the pinned source. However, NVIDIA's CUPTI
`GRAPH_TRACE` records a graph execution *without necessarily listing
the individual node executions*. The previously verified NVTX->CUPTI
kernel attribution covers the 1,664 recognized kernel records, not
necessarily all replayed graph nodes.

A new standalone read-only `gemma3_phase28_graph_audit` module
examines the existing SQLite export without GPU execution. It links
each runtime `cudaGraphLaunch` API interval to exactly one NVTX
phase; separately it requires each CUPTI GRAPH_TRACE correlationId to
resolve to a unique driver (preferred) or runtime graph-launch API
and the *same* phase. In the current single-GPU experiment it expects
one graph trace per runtime launch, and matching phase counts. If the
capture schema, correlation metadata or count relationship differs,
it reports **unverified** and requires manual schema inspection.
It never manufactures per-node graph replay coverage evidence.

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_context_dispatch \
  tests.test_phase28_gemma3_graph_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_graph_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite \
  --candidate bf16-on --n-ctx 2048
```

The audit reports `runtime_graph_launches_by_phase`,
`graph_trace_launches_by_phase`, driver/runtime correlation coverage,
the actual graph trace table columns, and counts of distinct graph
and executable IDs where the export supports them.
`cuda_graph_phase_attribution_verified=true` is **narrow**: it
confirms graph execution launches are attributable to NVTX ranges;
`graph_nodes_replayed_and_correlated_verified=false` and
`complete_attention_workload_coverage_verified=false` remain.
No profile is promoted to production.

## Hypothesis: mixed CUDA Graph replay and directly launched attention

The **first actual Phase28 Graph audit** of
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite` succeeded with
`cuda_graph_phase_attribution_verified=true`:
**128** CUDA Graph launches, 128 GRAPH_TRACE rows and 128 unique runtime
correlation matches. No ambiguous/unattributed graph replay was found.

| Phase | NVTX inference-call ranges | CUDA Graph launches | Difference |
|---|---:|---:|---:|
| Prefill | 96 | 68 | 28 |
| Decode | 64 | 60 | 4 |

Remarkably, the previous directly correlated Flash kernels are
`728 = 28 × 26` main prefill MMA_F16 launches and
`104 = 4 × 26` main decode VEC launches. Their helper kernel counts
also match exactly. This strongly supports the hypothesis that the
28 prefill ranges and four decode ranges without a Graph launch each
execute a direct kernel path for the 26 attention layers. The counts
by themselves do **not** prove a one-to-one assignment to ranges.

To test the hypothesis from the existing trace without CUDA execution,
a dedicated offline module now joins **each** individual NVTX CPU range
to `cudaGraphLaunch` runtime APIs and, independently, each observed
Flash kernel to its own CUDA runtime launch. It requires each range
to contain exactly one Graph launch and no directly correlated
Flash kernels, or zero Graph launches and exactly 26 main plus 26
auxiliary directly correlated Flash kernels. Unexpected/missing/mixed
patterns remain unverified. This 26-layer constraint is specific to the
pinned Gemma3-1B-IT workload.

```bash
.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_execution_paths -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_execution_paths \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite \
  --candidate bf16-on --n-ctx 2048
```

The result is deliberately limited to
`execution_path_classification_verified`. It cannot prove that
Nsight exposed **all internal CUDA Graph node executions**:
`cuda_graph_node_coverage_verified=false` and
`complete_attention_workload_coverage_verified=false` are always
retained. Neither Phase28 profiling nor this reader changes
production runtime settings.

## Per-NVTX execution paths verified — BF16 Flash ON, n_ctx=2048

The user's **actual** offline analysis of
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite` with
`ruth_runtime.gemma3_phase28_execution_paths` returned
`status="cuda_graph_and_direct_launch_paths_verified"` and
`execution_path_classification_verified=true`.

| Path | Prefill NVTX ranges | Decode NVTX ranges | Total |
|---|---:|---:|---:|
| CUDA Graph replay | 68 | 60 | 128 |
| Direct FlashAttention kernel launches | 28 | 4 | 32 |
| **NVTX ranges accounted for** | **96** | **64** | **160** |

The tool resolved 128 Graph replay executions through the SQLite
`GRAPH_TRACE` evidence and correlated **832 main + 832 auxiliary**
FlashAttention kernel launch records to the 32 direct-call ranges.
In the direct ranges, each of the 26 attention layers corresponds to
one main and one auxiliary launch. No mixed graph/direct ranges,
unknown ranges, unexpectedly sized direct paths or ambiguous direct
kernel launch correlations were reported:

- `graph_trace_launches_correlated=128`
- `direct_main_kernel_records_correlated=832`
- `direct_auxiliary_kernel_records_correlated=832`
- `direct_kernels_unattributed_or_ambiguous=0`
- `ranges_with_both_graph_and_direct_attention=0`
- `ranges_without_dispatch_evidence=0`
- `ranges_with_unexpected_direct_kernel_counts=0`

This elevates the previous *count-based hypothesis* to a verified
**per-range host launch-path classification** for the pinned
Gemma3-1B-IT BF16/BF16 Flash ON Phase28 benchmark. The separately
correlated direct main kernels are MMA_F16 for prefill and BF16-native
VEC for decode. **Do not extrapolate these exact kernels to each
replayed CUDA Graph node without further evidence.**

The analysis continues to report
`cuda_graph_node_coverage_verified=false`,
`complete_attention_workload_coverage_verified=false` and
`production_qualified=false`. The source SQLite export exposes
`CUPTI_ACTIVITY_KIND_KERNEL.graphId`, `graphNodeId` and the
`CUDA_GRAPH_EVENTS` table, which should be inspected read-only
before designing further graph-node coverage validation. The fact
that those fields exist is **not** proof of their completeness.

## Nsight 2025.6.3 default graph-level capture limits

User inspection of the verified Phase28 NVTX3 CUDA Graph trace
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3.sqlite` found:

- `CUPTI_ACTIVITY_KIND_KERNEL`: 24,120 records, **all** with
  `graphId=NULL` and `graphNodeId=NULL`.
- `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`: 128 records.
- `CUDA_GRAPH_EVENTS`: 34 records.

This does **not** contradict the independently verified 128
`cudaGraphLaunch` calls or their 68 prefill / 60 decode phase
correlations. The NVIDIA Nsight Systems guide documents that
`--cuda-graph-trace=graph` captures graph executions as a whole
and does not collect individual graph-node activities.
With a supported CUDA driver this is the default mode.

To collect **node-level** evidence, run a separate, explicitly
authorized Phase28 profiling experiment using
`--cuda-graph-trace=node`; keep the earlier graph-level SQLite
unchanged. Example on the user's Nsight 2025.6.3 installation:

```bash
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --sample=none --cpuctxsw=none \
  -o .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes \
  env RUTH_PHASE28_NVTX3_BRIDGE=1 \
  .venv/bin/python -m ruth_runtime.gemma3_phase28 \
    --candidate bf16-on --n-ctx 2048 \
    --execute --allow-gpu-inference

nsys export --type sqlite \
  --output .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.nsys-rep
```

Node tracing can introduce **substantial overhead**. Do not compare
the resulting prefill/decode wall times to graph-level runs as though
they were equivalent benchmark conditions. The previous graph-level
`gemma3_phase28_graph_audit` and `gemma3_phase28_execution_paths`
require `GRAPH_TRACE` and are **not** expected to validate a node-level
capture without schema-specific changes. Before modifying any audit,
inspect the new SQLite kernel rows, `graphId`, `graphNodeId`,
CUDA runtime launch correlations and phase markers in read-only mode.
Do not assert complete graph-node coverage from non-null IDs alone.
Production qualification remains false.

## CUDA Graph node-level capture: 81,008 kernel records

The user's new **node-level** Nsight Systems capture
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite`
reports **105,128 CUDA kernel activity records**, including:

| graphId | kernel records | replay-count consistency candidate |
|---|---:|---|
| 2 | 54,128 | 68 × 796 |
| 10 | 26,880 | 60 × 448 |
| NULL | 24,120 | direct launches; same as prior graph-level capture |

Both graphId and graphNodeId are populated for **81,008** kernel
records, with **1,244 distinct non-null graphNodeId** values overall.
The exact factors 796+448=1,244 agree with the previous run's
68 prefill and 60 decode graph replays. **A different Nsight capture
is a different process; graph IDs must not be assumed stable across
capture sessions.** Likewise, these aggregate equalities are
hypotheses until *per-node* frequencies are checked.

A new read-only offline module checks every observed pair
`(graphId, graphNodeId)` and its actual launch frequency.
It reports minimum, maximum and a frequency histogram per graph;
counts runtime `cudaGraphLaunch` calls inside NVTX CPU ranges from
the **same new node-level capture**; and inspects each graph's
FlashAttention kernel-name families. The potential graph→phase
correspondence is labelled `graph_phase_candidates_by_count`
and is **not treated as a direct causal join**.

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_node_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_node_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048
```

The status `observed_graph_kernel_node_repetition_counts_verified`
means only that the **observed CUDA kernel nodes** have complete,
uniform repetition counts consistent with the runtime graph launch
counts inside the capture. It is **not** a guarantee that graph
events, copies, memory sets, host functions or untraced node types
were fully captured. Hence:

- `graph_to_nvtx_phase_verified=false` until a genuine join exists;
- `all_cuda_graph_node_types_traced_verified=false`;
- `complete_attention_workload_coverage_verified=false`;
- `production_qualified=false`.

The pinned source/NVIDIA CUPTI docs support node-level kernel
`graphId` and `graphNodeId` fields. The node-level audit does not
execute CUDA or change any production runtime profiles.

## Uniform kernel-node replay confirmed on actual Phase28 node trace

The user ran the read-only node audit on
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite`.
It returned `observed_graph_kernel_node_repetitions_verified=true`,
with no partial graph/node identifiers and no ambiguous host
`cudaGraphLaunch` correlations:

| Graph ID | Distinct kernel node IDs | Observed repetitions per node | Kernel records |
|---|---:|---:|---:|
| 2 | 796 | 68 | 54,128 |
| 10 | 448 | 60 | 26,880 |
| No graph ID | n/a | n/a | 24,120 |

Each observed node in graph 2 appeared **exactly 68 times**, and
each node in graph 10 **exactly 60 times**. The main/helper Flash
families appear for exactly 26 attention layers *per replay*:

- Graph 2: `FLASH_MMA_F16=1,768=26×68`,
  `FLASH_FIXUP=1,768=26×68`;
  other CUDA kernels 50,592 = 744×68.
- Graph 10: `FLASH_VEC=1,560=26×60`,
  `FLASH_COMBINE=1,560=26×60`;
  other CUDA kernels 23,760 = 396×60.

These counts are compatible with 68 prefill CUDA Graph launches
and 60 decode CUDA Graph launches, and correspond to the 26-layer
direct-attention baseline. **Previous node-audit outputs explicitly
said `graph_to_nvtx_phase_verified=false`** because mapping a graph
ID to prefill or decode solely from equal replay counts is not
causal correlation.

The node audit has now been extended to make an *optional, direct*
`CUPTI_ACTIVITY_KIND_KERNEL.correlationId` →
`CUPTI_ACTIVITY_KIND_RUNTIME` `cudaGraphLaunch` →
same-thread NVTX interval join **within the same new SQLite capture**.
When present, this verifies a bijection between each unique graph
launch and a full set of the graph's observed kernel-node IDs,
with no missing/duplicate node in any replay, and unambiguous
graph-ID→NVTX-phase mapping. When kernel-node correlation IDs are
missing or differ in meaning on this Nsight version, this
**deliberately fails closed**; aggregate count coincidences still
cannot promote `graph_to_nvtx_phase_verified`.

To run the amended tests and re-evaluate the existing node trace
without any CUDA execution:

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin feat/gemma3-phase28-context-dispatch

.venv/bin/python -m unittest \
  tests.test_phase28_gemma3_node_audit -v

.venv/bin/python -m ruth_runtime.gemma3_phase28_node_audit \
  --sqlite .ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite \
  --candidate bf16-on --n-ctx 2048
```

Whether or not that optional correlation succeeds,
`all_cuda_graph_node_types_traced_verified=false`,
`complete_attention_workload_coverage_verified=false`,
and `production_qualified=false` remain. Only the individual
**observed CUDA kernel-node** records are covered.

## Actual Graph-node-to-NVTX causal attribution verified (BF16/ON, context 2048)

The user ran the revised `gemma3_phase28_node_audit` against the
existing node-level Nsight 2025.6.3 export
`.ruth-build/phase28/bf16-on-ctx2048-nvtx3-nodes.sqlite`:

- All **4** amended offline tests passed (`Ran 4 tests ... OK`).
- `graph_to_nvtx_phase_verified=true`.
- `graph_kernel_launch_replay_node_sets_verified=true`.
- `graph_kernel_launch_correlated_records=81008`.
- `graph_kernel_launch_unmatched_or_ambiguous_records=0`.
- `graph_kernel_launch_correlation_available=true`.
- `observed_graph_kernel_node_repetitions_verified=true`.
- `unmatched_or_ambiguous_graph_launches=0`.

The report establishes a **same-export causal association** via the
recorded kernel `correlationId` -> CUDA runtime
`cudaGraphLaunch` -> enclosing same-thread NVTX phase (it does not
infer this association from replay frequencies):

| Graph ID in this capture | Verified NVTX phase | Kernel node IDs | Replays | Kernel activity records |
|---|---|---:|---:|---:|
| 2 | prefill | 796 | 68 | 54,128 |
| 10 | decode | 448 | 60 | 26,880 |

For **each** of the 128 Graph launches, the reader verified the exact
set of observed kernel-node IDs appeared once per replay, and every
pair `(graphId, graphNodeId)` appeared 68 or 60 times respectively.
Graph 2 includes 1,768 MMA_F16 and 1,768 fixup kernel records; Graph
10 includes 1,560 BF16-native VEC and 1,560 combine records.
Dividing by replay counts gives 26 main + 26 helper attention kernel
records for each graph replay, matching the pinned model's 26
attention layers.

Combining this evidence with the previously verified **direct**
kernel-path analysis yields consistent main-attention accounting:

| Phase | Direct main Flash kernels | Graph main Flash kernels | Total | Expected per 26-layer call |
|---|---:|---:|---:|---:|
| prefill | 728 | 1,768 | 2,496 | 96 × 26 |
| decode | 104 | 1,560 | 1,664 | 64 × 26 |

The records and per-replay node checks support complete accounting
of the **observed main FlashAttention kernel invocations** for this
specific experimental workload, subject to the definitions and trace
coverage of Nsight Systems. This is **not** an assertion that every
possible type of CUDA Graph node was captured. In particular, the
reader still deliberately reports
`all_cuda_graph_node_types_traced_verified=false`,
`complete_attention_workload_coverage_verified=false`, and
`production_qualified=false`. Do not promote a production runtime
on this basis. The broader Phase28 candidate/context matrix and
numerical-quality assessments remain separate gates.

## Promotion gate

Phase 28 produces *diagnostic* context, timing, memory and CUDA-kernel
evidence, never a production policy selection. Longer-context numerical
accuracy, repeated workload timing, per-phase kernel attribution and
headroom remain open until their native evidence is inspected.
