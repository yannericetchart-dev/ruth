# Phase 28 — Gemma 3 CUDA dispatch evidence and long-context scaling

**Studio successivo sulle precisioni effettive:** [BF16/F16/F32, inventario dei dtype intermedi e fattibilità MMA BF16 nativo](gemma3-bf16-f16-f32-precision-paths-and-mma-options.md). Documento di ricerca, nessuna promozione alla produzione.

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

## Actual Phase 28 native result — F32/F32, Flash OFF, n_ctx=256

The user executed one explicitly authorized Gemma 3 1B IT BF16
GPU-inference experiment:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate f32-off --n-ctx 256 \
  --execute --allow-gpu-inference
```

**Observed report:**
`status=native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
`offloaded_layers=27`, `total_layers=27`,
`n_ctx_effective_native=256`,
`type_k=F32`, `type_v=F32`, Flash OFF and
`decode_logits_requested_per_token=true` with
`decode_logits_requested_count=16`.

| Observed metric | Value | Scope |
|---|---:|---|
| Prefill tokens | 128 | Two batches of 64 |
| Decode calls | 16 | Teacher-forced, logits on every call |
| Prefill median | **30.9847 ms** | Three timed repetitions |
| Prefill rate | **4,131.07 tokens/s** | 128 tokens / prefill median |
| Median individual decode latency | **12.592 ms/token** | Across all per-token calls |
| Median full decode round | **202.3512 ms** | Three 16-token rounds |
| Decode rate | **79.07 tokens/s** | 16 tokens / round median |
| CUDA model buffer | **1,907.39 MiB** | Native log |
| CUDA K/V buffers | **2 + 11 = 13 MiB** | Native log; matches logical KV plan |
| CUDA compute buffer | **64.28 MiB** | Native log |
| CPU_Mapped model buffer | **576 MiB** | Not an independently measured GPU allocation |
| Sampled GPU process peak | **2,100 MiB** | NVML sampling, not guaranteed instantaneous maximum |

The three prefill timing observations were
`34.569430`, `30.984712` and `30.944168` ms.
The selected pinned GGUF SHA256 was
`b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`
and pinned native wheel SHA256
`ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.
The native logits artifact has dtype `float32`,
shape `[2, 262144]`, local relative path
`.ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-off_k-F32_v-F32_hxddceoy.npz`
and SHA256
`93e993e1e1b9d72024044481727602486a2c8b03392a62ff8177fb71dcf2464e`.

**Evidence boundary:** `nvtx_ranges_available=false`,
`native_kernel_dispatch_verified=false`,
`production_qualified=false`, and
`runtime_profile_published=false`. This verifies
the requested native KV types and actual CUDA
teacher-forced inference, not the precise GPU kernel
instruction path or all-position numerical quality.
F32 *KV storage* is not an all-F32 computational
reference or mathematically exact ground truth.
The report contains two logits snapshots, not all
128 prefill or 16 decode positions.

**Next controlled comparison:** `mixed-on`, K=BF16/V=F16,
Flash ON, `n_ctx=256`, must run separately with
explicit `--execute --allow-gpu-inference`; do not
infer a result in advance. Numerical comparisons
must use `gemma3_phase28_compare` with a previously
verified same-context artifact and the corresponding
token-sequence equality check. Mixed-on vs BF16-on
isolates V-cache dtype for this fixed experimental
workload, subject to the actual backend dispatch.
No silent context increase, retry, promotion, or
change to the pinned native stack.

## Actual Phase 28 native result — BF16/F16, Flash ON, n_ctx=256

The user executed the explicitly authorized `mixed-on` Gemma 3
1B IT BF16 GPU experiment:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate mixed-on --n-ctx 256 \
  --execute --allow-gpu-inference
```

Result: `status=native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
`offloaded_layers=27`, `total_layers=27`,
`n_ctx_effective_native=256`,
`type_k=BF16`, `type_v=F16`,
`flash_attention=on`, `swa_full=true`
and `decode_logits_requested_count=16` with
`decode_logits_requested_per_token=true`.

| Observed metric | Value | Scope |
|---|---:|---|
| Prefill tokens/chunks | 128 / 2 | `n_batch=n_ubatch=64` |
| Decode calls | 16 | Teacher-forced, full-vocab logits each |
| Prefill median | **30.7740 ms** | Three timed repetitions |
| Prefill rate | **4,159.36 tokens/s** | 128 / median duration |
| Median individual decode latency | **12.4773 ms/token** | Per-call median |
| Median full decode round | **200.2478 ms** | Three 16-token rounds |
| Decode throughput | **79.90 tokens/s** | 16 / median round duration |
| CUDA model buffer | **1,907.39 MiB** | Native log |
| CUDA KV buffer | **1 + 5.5 = 6.5 MiB** | Native log, agrees with source-derived logical KV |
| CUDA compute buffer | **64.28 MiB** | Native log |
| CPU_Mapped model buffer | **576 MiB** | Not automatically additional VRAM |
| GPU process peak sampled | **2,100 MiB** | Sampling, not instantaneous max |

The three prefill observations were `33.327650`,
`30.557923`, `30.773997` ms.
The immutable prompt ID hash was
`bab4fb14dabe7263233449157076e3cfd1c317805de472c0bcf4208d1b8cad6c`,
matching the `f32-off` ctx256 run. The 16 decode IDs in both
reported runs were identical. Pinned GGUF SHA256
`b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`,
native wheel SHA256
`ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.

The local mixed logits artifact is
`.ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-on_k-BF16_v-F16_2g3xd9db.npz`,
SHA256
`69ed0413b8cb9de5472dae56297349d16fd5ba19ae07ea7086c2caeffe30ce16`,
dtype `float32`, shape `[2,262144]`.
It captures only last-prefill and last-decode
full-vocabulary logits. No full-token numerical fidelity
or production approval is implied.

### Completed four-configuration ctx256 slice

Earlier user-provided measured controls plus the two
just-observed native runs give:

| Candidate | K/V | Flash | Prefill median (ms) | Decode tok/s | Logical KV (MiB) |
|---|---|---|---:|---:|---:|
| `f16-off` | F16/F16 | OFF | 31.2959 | 79.46 | 6.5 |
| `bf16-on` | BF16/BF16 | ON | 30.8938 | 80.20 | 6.5 |
| `f32-off` | F32/F32 | OFF | 30.9847 | 79.07 | 13.0 |
| `mixed-on` | BF16/F16 | ON | 30.7740 | 79.90 | 6.5 |

**Four of four ctx256 cases have verified CUDA model
evaluation; ten of the sixteen Phase28 candidate/context
cases are now known native successes** (the previously
verified eight F16-OFF/BF16-ON long-context runs
plus these two new cases). This is not an assertion
that numerical-equivalence comparisons have completed
for all four candidates.

The most controlled *cache precision* contrast in
this slice is `bf16-on` vs `mixed-on`: Flash ON
and K=BF16 are fixed, V changes BF16→F16.
Kernel selection or temporary conversions may still
change; therefore an attribution of performance
entirely to V storage is not established. The
`f16-off` vs `f32-off` pair keeps Flash OFF
but changes both K and V. Comparing
`f32-off` to `mixed-on` changes Flash as well
and is confounded.

### Required offline numerical comparisons before further context escalation

Use `gemma3_phase28_compare` with previously
verified same-context local artifacts; the reader
checks exact prompt/decode token arrays and the
shape of both logits arrays and does not run CUDA.

**Flash ON, isolate V-cache type:**

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_compare \
  --baseline .ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-on_k-BF16_v-BF16_tfujrtrn.npz \
  --candidate .ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-on_k-BF16_v-F16_2g3xd9db.npz
```

**Flash OFF, compare F16/F16 to F32/F32 cache storage:**

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_compare \
  --baseline .ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-off_k-F16_v-F16_xapq63id.npz \
  --candidate .ruth-build/phase28/gemma-3-1b-it/ctx-256_flash-off_k-F32_v-F32_hxddceoy.npz
```

Both baselines are prior user-reported Phase28
ctx256 artifacts; the compare CLI independently
validates that both inputs have matching exact
token sequences and shape. Passing this gate
must not be mistaken for complete numerical
fidelity at unrecorded token positions.
Both exact offline comparisons have now **completed successfully**;
see the measured logit metrics and limitations in the section below.
Do not infer results for unmeasured positions.

For this `mixed-on` experiment:
`native_kernel_dispatch_verified=false`,
`nvtx_ranges_available=false`,
`production_qualified=false` and
`runtime_profile_published=false`.
No new Nsight collection is required for the
planned offline comparisons.

## Observed same-context numerical comparisons — ctx256 (both successful)

The user executed the two read-only comparisons between already
verified Phase28 Gemma 3 1B IT BF16 logits artifacts.
Both returned `status=full_logits_numeric_comparison`,
`same_teacher_forced_tokens=true`, `n_ctx=256`,
`vocabulary=262144`, `top1_match_fraction=1.0`,
`reference_precision_is_ground_truth=false`,
`native_kernel_dispatch_verified=false` and
`production_qualified=false`.

The numerical sample is **exactly two** full-vocabulary
F32 logit vectors per experiment: the last prefill position
(index 0) and last teacher-forced decode position (index 1).
It is *not* all intermediate positions of the 128-token
prefill or all 16 decode calls. Top-1=1.0 means 2/2
**recorded positions**, not general generation
equivalence.

| Metric | BF16/BF16 → BF16/F16 (Flash ON) | F16/F16 → F32/F32 (Flash OFF) |
|---|---:|---:|
| Jensen–Shannon mean (nats) | 8.647140722172348e-07 | 3.3662174115573566e-07 |
| KL baseline→candidate mean (nats) | 3.450529555342461e-06 | 1.3477870552100879e-06 |
| KL candidate→baseline mean (nats) | 3.4672603925615703e-06 | 1.3452012525382377e-06 |
| RMS logits difference | 0.012705533198433759 | 0.011415320492665115 |
| Mean absolute logits difference | 0.010114680961265776 | 0.009012389704039947 |
| Maximum absolute logits difference | 0.062470436096191406 | 0.05967235565185547 |
| Top-1 agreement | 2/2 | 2/2 |

**Per-position KL baseline→candidate:**

| Position | BF16/BF16 → BF16/F16 | F16/F16 → F32/F32 |
|---|---:|---:|
| Last prefill | 6.897949821766372e-06 | 2.6916049517113536e-06 |
| Last decode | 3.1092889185496652e-09 | 3.969158708822265e-09 |
| Maximum absolute difference at prefill | 0.062470436096191406 | 0.05967235565185547 |
| Maximum absolute difference at decode | 0.06228673458099365 | 0.0431060791015625 |

**Verified artifact identities — Flash ON, V storage changed:**

- Baseline K=BF16/V=BF16:
  `ctx-256_flash-on_k-BF16_v-BF16_tfujrtrn.npz`,
  SHA256
  `7741651087b5b6f3c1ecc57884c58efd4189c03c601d575f5c5065577733b84b`.
- Candidate K=BF16/V=F16:
  `ctx-256_flash-on_k-BF16_v-F16_2g3xd9db.npz`,
  SHA256
  `69ed0413b8cb9de5472dae56297349d16fd5ba19ae07ea7086c2caeffe30ce16`.

**Verified artifact identities — Flash OFF, both K and V storage changed:**

- Baseline K=F16/V=F16:
  `ctx-256_flash-off_k-F16_v-F16_xapq63id.npz`,
  SHA256
  `fa812cf2dd909729910981de78217fcc1dd3de9bdafdb3677c490495fd208de2`.
- Candidate K=F32/V=F32:
  `ctx-256_flash-off_k-F32_v-F32_hxddceoy.npz`,
  SHA256
  `93e993e1e1b9d72024044481727602486a2c8b03392a62ff8177fb71dcf2464e`.

### Numerical interpretation and remaining limits

Both experiments show very small mean
probability-distribution divergences on their
**two sampled token positions**, consistent
Top-1 labels, and nonzero raw logit differences
(up to about 0.06). The larger prefill KL than
decode KL in both comparisons is a fact of
these samples, not proof that other decode
positions always have negligible error.

The two comparisons are **not mutually ranked
for fidelity**: BF16/BF16 vs BF16/F16 changes only
V storage with Flash ON, while F16/F16 vs F32/F32
changes both K and V storage with Flash OFF.
Neither uses a true all-F32 model, and no
exact numerical ground truth is present.
Cross-pair JS or KL magnitudes are not a
quality score against a shared reference.

**Phase28 status:** the ctx256 four-candidate
native-inference slice is complete and these
two prioritized offline numeric checks are
complete. After the subsequently observed F32/F32 and
BF16/F16 ctx512 successes (see following
experimental reports), **four of the sixteen**
candidate/context pairs remain without
Phase28 native validation: `mixed-on` and
`f32-off` at ctx1024 and ctx2048.
The baseline `f16-off`/`bf16-on` runs at
those contexts have been reported previously.
Continuation requires a separately authorized
one-at-a-time GPU run, respecting the existing
native preflight and no-fallback requirements.

**Next recommended Phase28 gate:** compare the
newly verified ctx512 BF16/BF16 and BF16/F16
Flash-ON logits snapshots offline, holding
Flash and K fixed; use the existing exact
same-context artifact/token checks.
The ctx512 native experiments are complete.
Ctx512 uses 384 prefill tokens and 16 decode
tokens: absolute prefill latencies are not
directly comparable to ctx256's 128-token
workload. No automatic production promotion
or context escalation is authorized.

## Actual Phase 28 native result — F32/F32, Flash OFF, n_ctx=512

The user authorized and executed exactly one additional GPU experiment:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate f32-off --n-ctx 512 \
  --execute --allow-gpu-inference
```

**Observed status:** `native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
27/27 layers offloaded, `n_ctx_effective_native=512`,
K=F32, V=F32, Flash OFF,
`decode_logits_requested_per_token=true` and
`decode_logits_requested_count=16`.

| Metric | F32/F32 Flash OFF ctx256 | F32/F32 Flash OFF ctx512 |
|---|---:|---:|
| Prefill tokens/chunks | 128 / 2 | **384 / 6** |
| Decode calls | 16 | **16** |
| Prefill median ms | 30.9847 | **85.1810** |
| Prefill tokens/s | 4,131.07 | **4,508.05** |
| Decode median ms/token | 12.5920 | **12.9451** |
| Median 16-step decode total ms | 202.3512 | **207.4605** |
| Decode tokens/s | 79.07 | **77.12** |
| Logical/native KV MiB | 13 / 13 | **26 / 26** |
| Sampled GPU process peak MiB | 2,100 | **2,112** |

**Additional ctx512 native buffers:** CUDA model
`1907.39 MiB`, K/V `4.0 + 22.0 = 26.0 MiB`,
compute `64.28 MiB`, `CPU_Mapped` model
`576.0 MiB` (not separately counted as a
GPU memory buffer). Native type and context
allocation checks passed. Sampled peak can
miss instantaneous high-water events.
Prefill observations were `85.181024`,
`84.293824`, `87.115963` ms.

Relative to ctx256, reported prefill throughput
increased by about 9.1% while decode throughput
decreased by about 2.5%. These are **different
token-length workloads**, so the observation is
not a controlled context-length-only causal
effect. GPU peak samples differ by 12 MiB
while the logical KV buffer difference is
13 MiB; do not infer exact physical allocator
overhead from this sampled discrepancy.

**Pinned identities:** GGUF SHA256
`b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`,
native wheel SHA256
`ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`,
`prompt_ids_sha256=98bba70664e27cd85e8c566150f269e7cc1326206c40740566c63c077f2d1565`.
All 16 fixed decode IDs matched the established
experiment sequence.

**Verified local logits artifact:** F32
`[2,262144]` vectors at the final prefill and
final decode positions:
`.ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-off_k-F32_v-F32_kilmv137.npz`,
SHA256
`dd3d52eb5a2baa4a37f867cc16faa134fffa38d97525b1d8e4585ad973448a65`.

No numerical comparison at ctx512 has yet
been performed in this result; comparing to a
ctx256 artifact is invalid because the workloads
have different prompt coordinates. The new
artifact enables a **same-context** contrast
with the forthcoming `mixed-on` experiment,
whose Flash setting also differs, and with an
existing same-context `f16-off` artifact
once its exact file path is established.
`native_kernel_dispatch_verified=false`,
`nvtx_ranges_available=false`,
`production_qualified=false` and
`runtime_profile_published=false` remain.

**Historical matrix progress after this
F32-only test: 11/16 native successes.**
The BF16/F16 Flash ON ctx512 run has since
succeeded; see the next section for the
updated **12/16** status.
No automatic inference, context escalation
or production qualification followed.

## Actual Phase 28 native result — BF16/F16, Flash ON, n_ctx=512

The user executed both the read-only
`--candidate mixed-on --n-ctx 512` plan
(`status=read_only_plan`, no inference) and,
in a **separate explicitly authorized command**,
the native CUDA teacher-forced workload:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate mixed-on --n-ctx 512 \
  --execute --allow-gpu-inference
```

**Observed native outcome:**
`status=native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
`offloaded_layers=27`, `total_layers=27`,
`n_ctx_effective_native=512`,
`flash_attention=on`, `type_k=BF16`,
`type_v=F16`, `swa_full=true`;
`decode_logits_requested_per_token=true`
and `decode_logits_requested_count=16`.

| Metric | F32/F32 Flash OFF ctx512 | BF16/F16 Flash ON ctx512 |
|---|---:|---:|
| Prefill tokens / chunks | 384 / 6 | **384 / 6** |
| Decode tokens | 16 | **16** |
| Prefill median (ms) | 85.1810 | **83.4696** |
| Prefill throughput (tok/s) | 4,508.05 | **4,600.48** |
| Decode individual median (ms/token) | 12.9451 | **12.5837** |
| Decode round median (ms/16 tokens) | 207.4605 | **201.6221** |
| Decode throughput (tok/s) | 77.12 | **79.36** |
| Native/logical KV (MiB) | 26 / 26 | **13 / 13** |
| Sampled GPU peak (MiB) | 2,112 | **2,108** |

For the new BF16/F16 experiment, native CUDA
buffers were model `1907.39 MiB`,
KV `2.0 + 11.0 = 13.0 MiB`,
compute `64.28 MiB`; the additional
`CPU_Mapped` model buffer is `576.0 MiB`
and should not simply be added to VRAM.
Three observed prefill measurements:
`83.469583`, `83.108191`, `85.058622` ms.
Sampling of GPU peak utilization is not
a guaranteed instantaneous high-water mark.

The user's observed `f32-off` and
`mixed-on` ctx512 reports have matching
`prompt_ids_sha256`:
`98bba70664e27cd85e8c566150f269e7cc1326206c40740566c63c077f2d1565`;
the sixteen reported teacher-forced decode IDs
also match. The pinned GGUF SHA256 is
`b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`,
native wheel SHA256
`ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.

**New mixed-on logits artifact:** two float32
vectors, shape `[2,262144]`,
`.ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-on_k-BF16_v-F16_1ixlsmr0.npz`,
SHA256
`a60744b1d0935a674e580b5ba397d4e7d7589fb63221698f8df6d16202a88513`.
Native dispatch was not captured for this
run: `native_kernel_dispatch_verified=false`,
`nvtx_ranges_available=false`,
`production_qualified=false`,
`runtime_profile_published=false`.

This completes **12/16 Phase28 native experiments**:
all four configurations at n_ctx 256 and 512;
`f32-off` and `mixed-on` at n_ctx 1024 and
2048 remain untested in Phase28. The faster
reported prefill (+approximately 2.05%) and
decode (+approximately 2.90%) throughput of
mixed-on compared with f32-off **cannot be
attributed solely to cache precision**, since
K, V, Flash policy and possible dispatch
differ. The sample peak difference of
4 MiB is not an exact measurement of the
13 MiB logical KV difference.

### Next gate: ctx512 same-workload numerical checks (offline)

The stronger controlled pair is
**BF16/BF16 Flash ON vs BF16/F16 Flash ON**,
holding K=BF16 and Flash ON fixed and
changing only V's stored dtype. The
prior baseline `bf16-on` ctx512 artifact
was reported as
`.ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-on_k-BF16_v-BF16_3ctu967i.npz`
(SHA256
`2e8c8ca588ba2cb2f14fb24c5a0ce1ad385a76462f6adb433e42d58a64c2c5b6`).
The current report alone cannot revalidate
that previous local artifact; the offline
comparator checks the files and exact
prompt/decode token arrays.

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_compare \
  --baseline .ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-on_k-BF16_v-BF16_3ctu967i.npz \
  --candidate .ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-on_k-BF16_v-F16_1ixlsmr0.npz
```

A **secondary, confounded** pair compares
F32/F32 Flash OFF and BF16/F16 Flash ON.
Both reports have matching prompt/decode IDs,
but multiple independent configuration
variables change; their numerical distance
does not isolate V or Flash independently.

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28_compare \
  --baseline .ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-off_k-F32_v-F32_kilmv137.npz \
  --candidate .ruth-build/phase28/gemma-3-1b-it/ctx-512_flash-on_k-BF16_v-F16_1ixlsmr0.npz
```

Both ctx512 offline numerical comparisons
have now been executed successfully; the
exact observed KL/JS/RMS/Top-1 results and
evidence boundaries are recorded below.
Do not infer equivalence for all intermediate
tokens from the two recorded logits rows.
No new GPU inference was needed for
these offline comparisons.

## Actual Phase 28 offline numerical comparisons — n_ctx=512 (completed)

The user executed **both** `gemma3_phase28_compare`
commands against the previously measured ctx512
Gemma 3 1B BF16 logit artifacts.
Each report returned:
`status=full_logits_numeric_comparison`,
`same_teacher_forced_tokens=true`,
`n_ctx=512`, `vocabulary=262144`,
`top1_match_fraction=1.0`,
`reference_precision_is_ground_truth=false`,
`native_kernel_dispatch_verified=false` and
`production_qualified=false`.

The results concern only **two sampled
full-vocabulary F32 logits vectors per run**:
last prefill and last teacher-forced decode.
Top-1=1.0 means **2/2 sampled positions**,
not general functional equivalence,
continuous-token correctness or text
generation fidelity.

| Observed metric | BF16/BF16 → BF16/F16 (Flash ON) | F32/F32 Flash OFF → BF16/F16 Flash ON |
|---|---:|---:|
| Jensen–Shannon mean, nats | 1.6112830324160837e-07 | 1.1904699428559201e-08 |
| KL baseline→candidate mean, nats | 6.386638597335922e-07 | 4.77137146009051e-08 |
| KL candidate→baseline mean, nats | 6.505380650213616e-07 | 4.7527108356896585e-08 |
| RMS logit difference | 0.019654959610093344 | 0.018967132781074446 |
| Mean absolute logit difference | 0.015261355799196963 | 0.01300561648147891 |
| Maximum absolute logit difference | 0.10388755798339844 | 0.11905193328857422 |
| Top-1 matches | 2/2 | 2/2 |

**Per-position KL baseline→candidate:**

| Position | BF16/BF16 → BF16/F16 | F32/F32 OFF → BF16/F16 ON |
|---|---:|---:|
| Last prefill | 3.3369001611093054e-08 | 5.2371333653131703e-08 |
| Last decode | 1.2439587178560913e-06 | 4.3056095548678495e-08 |
| Maximum absolute logit diff, prefill | 0.10388755798339844 | 0.11905193328857422 |
| Maximum absolute logit diff, decode | 0.0962831974029541 | 0.03526020050048828 |

**Verified ctx512 artifacts and SHA256:**

- Controlled BF16/BF16 Flash ON baseline:
  `ctx-512_flash-on_k-BF16_v-BF16_3ctu967i.npz`,
  `2e8c8ca588ba2cb2f14fb24c5a0ce1ad385a76462f6adb433e42d58a64c2c5b6`.
- Shared mixed BF16/F16 Flash ON candidate:
  `ctx-512_flash-on_k-BF16_v-F16_1ixlsmr0.npz`,
  `a60744b1d0935a674e580b5ba397d4e7d7589fb63221698f8df6d16202a88513`.
- Secondary F32/F32 Flash OFF baseline:
  `ctx-512_flash-off_k-F32_v-F32_kilmv137.npz`,
  `dd3d52eb5a2baa4a37f867cc16faa134fffa38d97525b1d8e4585ad973448a65`.

### Interpretation and next matrix gate

The **controlled** BF16/BF16→BF16/F16
comparison holds K=BF16 and Flash ON
constant while changing the stored V
dtype. It reveals non-identical raw logits
but low divergences of the associated
probability distributions on the two sampled
positions and unchanged Top-1. Unlike the
ctx256 controlled pair, the sampled decode KL
at ctx512 (1.2439587178560913e-06) is
greater than the sampled prefill KL
(3.3369001611093054e-08). This reversal
must not be interpreted as a universal
decode-instability trend: the prompt and
relative positions change with n_ctx.

The **secondary** F32/F32 Flash OFF→
BF16/F16 Flash ON comparison changes
multiple variables simultaneously.
The lower KL/JS figures do **not**
constitute evidence that the mixed cache
is closer to mathematical truth: neither
run is a ground-truth reference and the
pairs have different baselines.
An absolute logit difference as high as
0.11905193328857422 can coexist with
small softmax KL/JS; nonzero differences
are not erased by Top-1 agreement.

The ctx512 numerical gate is complete for
the two chosen comparisons, subject to
the explicitly documented two-position
coverage. At that moment, **12/16** Phase28
candidate/context GPU experiments had
been verified (all four at 256 and 512).
A subsequent F32/F32 Flash OFF ctx1024
native result is now documented below and
advances the matrix to **13/16**.
All existing `production_qualified`,
`conversion_elimination_safe_verified`
and related CUDA-dataflow gates remain
unchanged.

## Actual Phase 28 native result — F32/F32, Flash OFF, n_ctx=1024

The user executed the read-only `f32-off` ctx1024
plan separately and then explicitly authorized
a **single** native CUDA experiment:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate f32-off --n-ctx 1024 \
  --execute --allow-gpu-inference
```

The worker returned
`status=native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
`offloaded_layers=27` / `total_layers=27`,
`n_ctx_effective_native=1024`, K=F32,
V=F32, Flash OFF and
`decode_logits_requested_per_token=true`
with `decode_logits_requested_count=16`.

| Metric | F32/F32 OFF ctx256 | F32/F32 OFF ctx512 | F32/F32 OFF ctx1024 |
|---|---:|---:|---:|
| Prefill tokens | 128 | 384 | **768** |
| Prefill chunks (`n_batch=64`) | 2 | 6 | **12** |
| Decode teacher-forced calls | 16 | 16 | **16** |
| Prefill median ms | 30.9847 | 85.1810 | **170.2965** |
| Prefill throughput, tokens/s | 4,131.07 | 4,508.05 | **4,509.78** |
| Decode individual median ms/token | 12.5920 | 12.9451 | **13.4313** |
| Decode median whole round, ms | 202.3512 | 207.4605 | **214.7660** |
| Decode throughput, tokens/s | 79.07 | 77.12 | **74.50** |
| Native/logical KV, MiB | 13 / 13 | 26 / 26 | **52 / 52** |
| NVML GPU process peak sampled, MiB | 2,100 | 2,112 | **2,138** |

Observed CUDA buffers: model `1907.39 MiB`,
KV `8.0 MiB + 44.0 MiB = 52.0 MiB`,
compute `64.28 MiB`; the `CPU_Mapped`
model buffer is `576.0 MiB`, not a separate
CUDA buffer that should be added mechanically
to the GPU peak sample. The three prefill
measurements were `168.673348`,
`170.829467` and `170.296532` ms.
GPU peak is sampled, not necessarily the
instantaneous maximum. Increasing context
also increased the fixed prefill workload,
so timing trends are not a context-only
causal estimate.

The recorded F32/F32 ctx1024 results
show prefill throughput practically
unchanged from ctx512 (4,509.78 vs
4,508.05 tokens/s) and decode throughput
lower than ctx256 (74.50 vs 79.07 tokens/s,
approximately 5.8%). KV scales as
source-derived and native allocation
measurements predict. No quantitative
precision or production superiority can
be inferred from this isolated experiment.

**Pinned identity and local numerical artifact:**

- GGUF SHA256:
  `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`.
- Native wheel SHA256:
  `ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.
- Prompt token IDs SHA256:
  `b3f6ab50b2d2f49739d85a35470c9d9ad5328753b0feecb075df4bf4491ddeec`.
- Logits: `float32`, shape `[2, 262144]`
  (last prefill and last decode full-vocabulary rows).
- Local path:
  `.ruth-build/phase28/gemma-3-1b-it/ctx-1024_flash-off_k-F32_v-F32_54m9floe.npz`.
- Artifact SHA256:
  `9f8bce9ae5ef51434c0d8fdb7aac75b513d76c7bf55e6808340badaf1e3ca30c`.

`native_kernel_dispatch_verified=false`,
`nvtx_ranges_available=false`,
`production_qualified=false` and
`runtime_profile_published=false`.
The two-position logits artifact is not
proof of long-sequence numeric equivalence.
Only compare it offline against a matching
ctx1024 artifact with an identical
teacher-forced token sequence, subject
to the strict existing compare gate.

**Historical matrix progress at the time of
this F32/F32 ctx1024 experiment: 13/16 native successes.**
The subsequent `mixed-on` BF16/F16 Flash
ON ctx1024 run has since succeeded and is
documented below, advancing to **14/16**.
The original next controlled step was the
**read-only** `mixed-on` ctx1024 plan:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate mixed-on --n-ctx 1024
```

This plan predicted 768 prefill tokens,
16 teacher-forced decode calls and
**26 MiB** of mixed BF16/F16 logical KV.
The read-only plan did not allocate or run
the model; the separately authorized
native test is documented below.
Do not auto-escalate to ctx2048,
alter the pinned native library or
promote a configuration to production.

## Actual Phase 28 native result — BF16/F16, Flash ON, n_ctx=1024

The user executed the read-only `mixed-on`
ctx1024 plan (`status=read_only_plan`,
no native execution) and then **separately
authorized exactly one** teacher-forced
CUDA evaluation:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate mixed-on --n-ctx 1024 \
  --execute --allow-gpu-inference
```

**Observed status:** `native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
`offloaded_layers=27` / `total_layers=27`,
`n_ctx_effective_native=1024`,
K=BF16, V=F16, Flash ON,
`decode_logits_requested_per_token=true`
and `decode_logits_requested_count=16`.

| Metric | F32/F32 Flash OFF ctx1024 | BF16/F16 Flash ON ctx1024 |
|---|---:|---:|
| Prefill tokens / chunks | 768 / 12 | **768 / 12** |
| Decode teacher-forced calls | 16 | **16** |
| Prefill median (ms) | 170.2965 | **160.6581** |
| Prefill throughput (tokens/s) | 4,509.78 | **4,780.34** |
| Median per-call decode (ms/token) | 13.4313 | **12.7386** |
| Median 16-token decode round (ms) | 214.7660 | **204.0292** |
| Decode throughput (tokens/s) | 74.50 | **78.42** |
| Native and logical KV (MiB) | 52.0 | **26.0** |
| NVML GPU process peak sampled (MiB) | 2,138 | **2,120** |

Native `mixed-on` CUDA buffers:
model `1907.39 MiB`,
KV `4.0 MiB + 22.0 MiB = 26.0 MiB`,
compute `64.28 MiB`;
`CPU_Mapped` model `576.0 MiB` is
not a separately measured CUDA VRAM
allocation. Three prefill observations:
`159.772191`, `160.658083`,
`161.753122` ms. NVML's 2120 MiB
GPU peak is a sampled observation and may
miss short allocation peaks.

Relative to F32/F32 Flash OFF at the
**same 768-token Phase28 workload**,
mixed-on showed about +6.0% prefill
and +5.3% decode throughput, and exactly
half the KV storage. This is **not a
cache-dtype-only causal comparison**:
K, V and Flash policy (and potentially
native kernel dispatch) change. The
18 MiB difference between GPU sampled
peaks must not be interpreted as an
exact allocation savings of 26 MiB.

**Pinned provenance and matched tokens:**

- GGUF SHA256:
  `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`.
- Native wheel SHA256:
  `ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.
- Prompt token IDs SHA256:
  `b3f6ab50b2d2f49739d85a35470c9d9ad5328753b0feecb075df4bf4491ddeec`,
  identical to the observed ctx1024
  F32/F32 run.
- The 16 reported teacher-forced
  decode IDs also match that run.
- Local logits artifact, dtype `float32`,
  shape `[2, 262144]` (last prefill
  and last decode full-vocabulary vectors):
  `.ruth-build/phase28/gemma-3-1b-it/ctx-1024_flash-on_k-BF16_v-F16_easavhiv.npz`.
- Artifact SHA256:
  `d93ec1476381685c029b5f50c0a5bca956eb0cb6670d3fdb6da1b18044f0ca58`.

`nvtx_ranges_available=false`,
`native_kernel_dispatch_verified=false`,
`production_qualified=false` and
`runtime_profile_published=false`.
A full-token numeric quality gate has
**not** been completed by this
native inference alone. Only
two positions' F32 logits were archived.

### Matrix gate — 14/16 native cases

At contexts 256, 512 and 1024,
all four Phase28 candidates have now
completed native GPU inference.
**Only `f32-off` and `mixed-on`
at `n_ctx=2048` remain untested.**

The immediate next step is an
**offline, same-context, identical-token**
numeric comparison. The strongest
controlled pair is prior BF16/BF16
Flash ON against newly observed
BF16/F16 Flash ON: fixed K=BF16,
Flash ON, only V storage changes.
The exact BF16/BF16 ctx1024
baseline filename was **not confirmed**
by this message, so do not invent it.

On the user's local machine, discover
the existing baseline by its exact
Phase28 case prefix and ensure that
there is **exactly one** matching file:

```bash
cd ~/Dev/ruth-model-profiling
(
  shopt -s nullglob
  baseline=(.ruth-build/phase28/gemma-3-1b-it/ctx-1024_flash-on_k-BF16_v-BF16_*.npz)
  if (( ${#baseline[@]} != 1 )); then
    printf 'Expected exactly one BF16/BF16 ctx1024 baseline; found %d\n' "${#baseline[@]}"
    exit 1
  fi
  .venv/bin/python -m ruth_runtime.gemma3_phase28_compare \
    --baseline "${baseline[0]}" \
    --candidate .ruth-build/phase28/gemma-3-1b-it/ctx-1024_flash-on_k-BF16_v-F16_easavhiv.npz
)
```

The CLI reads local artifacts and
validates matching context length,
exact prompt and decode arrays and
logits shapes. No CUDA execution
or production profile change is
authorized by this command.
The BF16/BF16 controlled comparison
has now **completed successfully**.
Its numerical JSON metrics, hashes and
limitations are recorded in the following
section. An additional confounded F32/F32
OFF vs BF16/F16 ON comparison has
**not** yet been performed at ctx1024;
it cannot isolate a single cache dtype
change, and is not required to complete
the controlled V-precision comparison.

The next ctx2048 GPU experiment
requires separate authorization
and success of the existing
native preflight. No automatic
context escalation, download,
fallback or promotion is approved.

## Verified Phase 28 controlled numerical comparison — n_ctx=1024

The user ran the read-only Phase28 logits comparator on
two local ctx1024 artifacts, after the guarded Bash
baseline glob returned exactly one BF16/BF16 Flash ON
file. The comparator independently reported
`status=full_logits_numeric_comparison`,
`same_teacher_forced_tokens=true`,
`n_ctx=1024`, `vocabulary=262144`,
`top1_match_fraction=1.0`,
`reference_precision_is_ground_truth=false`,
`native_kernel_dispatch_verified=false`, and
`production_qualified=false`.

**Controlled variables:** Gemma 3 1B IT BF16 GGUF,
same teacher-forced tokens and context length,
K=BF16, Flash ON; only stored V dtype changes
BF16→F16. Backend kernel selection and
temporary conversions may still differ.

**Observed artifacts:**

- Baseline BF16/BF16 Flash ON at ctx1024:
  local path prefix
  `.ruth-build/phase28/gemma-3-1b-it/ctx-1024_flash-on_k-BF16_v-BF16_*.npz`
  (exact basename is **not available** in the
  reported JSON). Its full-file SHA256 was
  `b59adb7dc00eb103b255d0ffbab8ab0425c71c868b2f1c85d48d4ea90261c167`.
- Candidate BF16/F16 Flash ON at ctx1024:
  `.ruth-build/phase28/gemma-3-1b-it/ctx-1024_flash-on_k-BF16_v-F16_easavhiv.npz`,
  SHA256
  `d93ec1476381685c029b5f50c0a5bca956eb0cb6670d3fdb6da1b18044f0ca58`.

**Observed numeric metrics, across two sampled
full-vocabulary logit positions (last prefill
and last decode):**

| Metric | Result |
|---|---:|
| Jensen–Shannon mean (nats) | 4.4264470887242736e-08 |
| KL baseline→candidate mean (nats) | 1.7772785404097928e-07 |
| KL candidate→baseline mean (nats) | 1.764449849155692e-07 |
| RMS logit difference | 0.028662088104959633 |
| Mean absolute logit difference | 0.023109153727887133 |
| Maximum absolute logit difference | 0.13928604125976562 |
| Top-1 agreement | 2/2 |

Per-position measurements:

| Position | KL baseline→candidate (nats) | Max absolute logit difference | Top-1 |
|---|---:|---:|---|
| Last prefill, index 0 | 9.557285614792214e-08 | 0.13928604125976562 | equal |
| Last decode, index 1 | 2.5988285193403645e-07 | 0.12633800506591797 | equal |

### Across-context V-precision experiment — native success and two-position metrics

| ctx | JS mean (nats) | KL baseline→candidate mean (nats) | RMS logits | Max abs logits | Top-1 |
|---:|---:|---:|---:|---:|---|
| 256 | 8.647140722172348e-07 | 3.450529555342461e-06 | 0.012705533198433759 | 0.062470436096191406 | 2/2 |
| 512 | 1.6112830324160837e-07 | 6.386638597335922e-07 | 0.019654959610093344 | 0.10388755798339844 | 2/2 |
| 1024 | 4.4264470887242736e-08 | 1.7772785404097928e-07 | 0.028662088104959633 | 0.13928604125976562 | 2/2 |

While raw-logit RMS and maximum
absolute differences increased in
these three samples, mean softmax
KL/JS divergences decreased. These
metrics have distinct mathematical
definitions, and different context
experiments have different prompt
lengths/relative token positions:
**do not infer monotonic improvements
in model accuracy or degradation in
long-sequence generation** from this
table. The final-decode KL exceeded
final-prefill KL by approximately
2.72× at ctx1024, whereas ctx256 and
ctx512 showed different prefill/decode
relationships. No intermediate logits
were saved, and the tests are
teacher-forced, not sampled text
generation. Top-1 2/2 at each context
is a narrow observation, not a
guarantee of sequence equivalence.

**Historical matrix status at this numeric check:**
14/16 native-successful candidate/context
cases (all four at ctx256, ctx512 and
ctx1024). At that time, the remaining
native experiments were `f32-off` and
`mixed-on` at ctx2048. The `f32-off`
ctx2048 native execution has **subsequently
succeeded** and is documented below,
updating progress to **15/16**. The
next safe step at the time was a
read-only `f32-off` ctx2048 plan:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate f32-off --n-ctx 2048
```

This plan predicted 1536 prefill
tokens (24 chunks of 64), 16
teacher-forced decode calls and
104 MiB logical F32/F32 KV;
the subsequently authorized native
run verified the matching native KV
buffers (see following section).
The as-yet unrun mixed BF16/F16
Flash ON ctx2048 case is predicted
to use 52 MiB logical KV; this
does **not** verify its native
buffers, GPU fit, execution or
numeric quality. No automatic
inference, downloads, CPU fallback,
pinned native library modification
or production promotion.

## Actual Phase 28 native result — F32/F32, Flash OFF, n_ctx=2048

The user first executed an isolated read-only plan
(`status=read_only_plan`, `inference_executed=false`)
and then **separately authorized** one native CUDA
experiment with the pinned Gemma 3 1B IT BF16 model:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate f32-off --n-ctx 2048 \
  --execute --allow-gpu-inference
```

**Observed:** `status=native_long_context_inference_observed`,
`inference_executed=true`,
`full_gpu_offload_observed=true`,
`offloaded_layers=27`, `total_layers=27`,
`n_ctx_effective_native=2048`, Flash OFF,
K=F32, V=F32,
`decode_logits_requested_per_token=true`,
`decode_logits_requested_count=16`.
`swa_full=true`, single-sequence workload,
`n_batch=n_ubatch=64` are unchanged.

### F32/F32 Flash OFF — observed context scaling

| Metric | ctx256 | ctx512 | ctx1024 | ctx2048 |
|---|---:|---:|---:|---:|
| Prefill tokens / chunks | 128 / 2 | 384 / 6 | 768 / 12 | **1536 / 24** |
| Teacher-forced decode calls | 16 | 16 | 16 | **16** |
| Prefill median, ms | 30.9847 | 85.1810 | 170.2965 | **342.3211** |
| Prefill throughput, tokens/s | 4131.07 | 4508.05 | 4509.78 | **4487.02** |
| Decode individual median, ms/token | 12.5920 | 12.9451 | 13.4313 | **14.1392** |
| Decode median total, ms per 16 tokens | 202.3512 | 207.4605 | 214.7660 | **226.6777** |
| Decode throughput, tokens/s | 79.07 | 77.12 | 74.50 | **70.58** |
| Logical/native KV, MiB | 13/13 | 26/26 | 52/52 | **104/104** |
| Sampled GPU process memory peak, MiB | 2100 | 2112 | 2138 | **2190** |

The native CUDA buffers were
`1907.39 MiB` model, two KV buffers
`16.0 + 88.0 = 104.0 MiB`,
and `64.28 MiB` compute.
The additional `CPU_Mapped` model buffer
was `576.0 MiB`, not an independently
measured GPU VRAM buffer. The native
KV sum agrees exactly with the pinned
source-derived 104.0 MiB logical estimate.

The three observed prefill times were
`342.321076`, `342.356966` and
`341.711720` ms. Prefill throughput
remained near 4500 tokens/s from ctx512
through ctx2048. Decode throughput
decreased from 79.07 tokens/s at ctx256
to 70.58 tokens/s at ctx2048, about
10.7% lower. The experiment changes
the prompt token count with context,
so do **not** attribute the whole
difference exclusively to configured
context size. NVML sampled peak
`2190 MiB` is not proof of the
instantaneous maximum; the ctx1024
comparison sample was `2138 MiB`.

### Provenance and numeric artifact

- Pinned GGUF SHA256:
  `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`.
- Native wheel SHA256:
  `ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.
- Prompt ID SHA256:
  `8cab7dc542f4f12f2b73989ddf29d0c32d87f8cbdd79e6f1b04b923526dfbc25`.
- Fixed teacher-forced decode IDs:
  `3656,1156,1806,2390,3493,3962,6819,6589,10155,3595,236761,3656,1156,1806,2390,3493`.
- Local full-vocabulary float32 logits:
  shape `[2,262144]`, saved at last
  prefill and last decode positions:
  `.ruth-build/phase28/gemma-3-1b-it/ctx-2048_flash-off_k-F32_v-F32_r7anfs8n.npz`.
- Artifact SHA256:
  `18401161b8460630ffb3f7744d76c9c099057239f7a681089ec46edf9e5988b2`.

No same-context numerical fidelity
comparison has yet been reported
for this new F32/F32 Flash OFF artifact.
`native_kernel_dispatch_verified=false`,
`nvtx_ranges_available=false`,
`production_qualified=false`,
`runtime_profile_published=false`.
The native test by itself neither
qualifies model accuracy nor proves
a particular kernel instruction path.

### Phase28 matrix status: 15/16

All four candidates have passed native
CUDA model evaluation at ctx256,
ctx512 and ctx1024. At ctx2048,
`f16-off`, `bf16-on` and now
`f32-off` have been verified;
**only `mixed-on` (BF16/F16 Flash ON)
ctx2048 remains untested**.

The next experiment starts with
a **read-only plan**, never an
implicit GPU run:

```bash
.venv/bin/python -m ruth_runtime.gemma3_phase28 \
  --candidate mixed-on --n-ctx 2048
```

Expected from the approved gate:
1536 prefill tokens, 24 chunks,
16 teacher-forced decode calls
and **52 MiB** source-derived
BF16/F16 logical KV. This is a
forecast, **not** native observation
for `mixed-on` ctx2048.

Once the plan is inspected, a
single native `mixed-on` test may
be run only under separately
explicit `--execute --allow-gpu-inference`
user authorization and with the
same pinned GPU preflight and
full-offload checks. On success,
use matching ctx2048 logits
artifacts for offline numerical
comparisons, most importantly
BF16/BF16 Flash ON vs BF16/F16
Flash ON (fixed K and Flash),
and secondarily F32/F32 Flash
OFF vs BF16/F16 Flash ON
(multiple variables changed).
No new CUDA profiler investigation,
auto-selection or production
promotion is implied by the
16-case experiment matrix.

## Promotion gate

Phase 28 produces *diagnostic* context, timing, memory and CUDA-kernel
evidence, never a production policy selection. Longer-context numerical
accuracy, repeated workload timing, per-phase kernel attribution and
headroom remain open until their native evidence is inspected.
