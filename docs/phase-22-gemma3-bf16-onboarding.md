# Phase 22: Gemma 3 1B IT BF16 — step-by-step new-model onboarding

## Target and principle

New production **candidate** (not automatically promoted to default):

- Model: `google/gemma-3-1b-it`, text-only Gemma 3 architecture.
- Exact file selected by user:
  [google_gemma-3-1b-it-bf16.gguf](https://huggingface.co/bartowski/google_gemma-3-1b-it-GGUF/blob/main/google_gemma-3-1b-it-bf16.gguf)
- Publisher: `bartowski/google_gemma-3-1b-it-GGUF`
- Immutable revision: `6de70cb020706f01529b27fee7ab17c5969041c8`
  (file upload commit; Ruth will verify that revision's complete metadata).
- SHA256: `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`
- Expected exact bytes: `2006573568`.
- License: `gemma`; explicit user review/acknowledgment required.
- GGUF weight storage format: `BF16` (bfloat16).
- Native backend: original **unpatched** `llama.cpp`
  `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- No universal reuse of Gemma 2's 21+21 KV layer geometry: Gemma 3
  architecture must be read from this GGUF and the pinned native backend.
- The actual GPU weight buffer size, CUDA initialization, context KV/compute
  bytes, Flash/SWA layer structure and CUDA fit are all **UNKNOWN** until
  independently verified. Do not infer them from the 2.01 GB file size.

## Correct floating format terminology

`BF32` is **not** a supported GGUF weight tensor type. Do not create a
fictional `BF32` file or claim that BF16 has F32's numerical precision.

- **BF16**: 16 bits, exponent 8, fraction 7. FP32-like exponent range,
  lower mantissa resolution, half F32 storage. Can use Tensor Cores when
  the backend/kernel supports BF16.
- **F16**: 16 bits, exponent 5, fraction 10, half F32 storage, potentially
  limited dynamic range with underflow/overflow. Tensor Core capable.
- **F32**: 32 bits, exponent 8, fraction 23. Memory-intensive.
  Some implementations may accelerate FP32 computations using Tensor Core
  **TF32** operations; TF32 is a compute mode and **does not halve F32
  storage**. Never claim hardware kernel acceleration without measurement.

Gemma 2 9B IT's **original** checkpoint was also exported in BF16,
not native F32 (Google's model card confirms this). Its full-precision
unquantized weights are simply too large for the current 6 GiB GPU;
the existing quantized Gemma 2 profile must not be deleted.

Ruth policy: all BF16/F16/F32 are unquantized quality class 1. The
user-selected operational tie-break is BF16 > F16 > F32, including
warnings for F16/F32. This is **not** a measured cross-model semantic
quality ranking. Prefer fidelity to the original checkpoint dtype.

## Staged steps — do not skip gates

### 0 — Catalog pin and canonical profile (COMPLETED IN GIT)

`catalog/model_catalog.json` now has BF16 Gemma3 with immutable
repository revision, filename, exact byte size and SHA256.
`profiles/models/gemma/gemma-3/gemma-3-1b-it/profile.json` is the
**only** canonical model profile. Geometry and native measurements are null
or empty, not copied from Gemma2.

The original production model remains as a registered fallback and is
not silently replaced before new-model validation.

### 1 — Source metadata (NEXT; no weights, no GPU)

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling

./ruth models select --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --format gguf \
  --quantization BF16

./ruth models resolve --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --format gguf \
  --quantization BF16 --allow-metadata-network

# Optional additional check: enumerate exact filenames at that same HF revision
./ruth models quantizations --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --format gguf \
  --quantization BF16 --allow-metadata-network
```

Required result: `source_metadata_verified_license_pending` with exact
SHA, size, revision, filename and license. Any mismatch stops progression
without download. This is a public metadata HTTP request only. An actual
Hugging Face Xet payload may require adapting Ruth's *verified* metadata
parser if its API fields have changed; do not bypass SHA verification.

### 2 — License and verified download (requires explicit user choice)

Review the Gemma license and grant authorization explicitly:

```bash
./ruth models download --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --format gguf \
  --quantization BF16 --allow-model-download --accept-model-license
```

No model is downloaded automatically as a result of step 1.
The downloader must independently verify SHA256, byte count and local
GGUF magic and promote only the matching file.

### 3 — Native library & backend gates (no unguarded fallback)

Verify the pinned wheel/native ABI, GPU UUID, single-GPU preflight,
architecture support and that the pinned llama.cpp contains Gemma 3
and BF16 weight tensor handling. Never swap to a different wheel without
a separate backend policy/profile revision.

### 4 — Weights-only CUDA (after stages 1–3)

```bash
./ruth models probe-load --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --format gguf \
  --quantization BF16 --load-mode weights \
  --allow-gpu-load
```

The worker now explicitly permits **only weights** for this novel model;
it refuses Gemma3 context/KV until the GGUF's architecture has been
independently inspected. Only actual measured CUDA weights buffers,
driver and CUDA overhead become known. Raw local evidence is preserved
and the owner merges results into the **same** canonical Git profile.

### 5 — GGUF metadata and tensor descriptor inspection (READY; read-only)

After successful model download and BF16 weights-only native probe,
inspect the same immutable local GGUF **without CUDA, native weight
allocation, tokenizer or any inference**:

```bash
git pull --ff-only origin model-profiling

./ruth models inspect-gguf --purpose production \
  --family gemma --model gemma-3 --version gemma-3-1b-it \
  --quantization BF16
```

The command first checks the verified download manifest and **SHA256 of
the entire pinned GGUF**. It then parses only the bounded GGUF metadata
header and tensor descriptors (not the tensor bytes). It requires
`general.architecture=gemma3`, a positive layer count, and BF16 as
the dominant storage type by **multidimensional tensor elements**.
`general.file_type` is optional advisory metadata, and value 24 is
`GGML_FTYPE_MOSTLY_BF16` in the pinned ggml revision. Missing or
different values are preserved as `declared_file_type` with explicit
`metadata_warnings`; they are NOT silently overwritten. Tensor
descriptors are checked separately. It reports
exact observed `gemma3.*` hyperparameters, BF16/F32/F16 tensor counts
and missing values without making up global/SWA classification.
It may encounter 1B checkpoint metadata that lacks some head fields.
Missing means UNKNOWN.

A successful read-only result is
`gguf_header_geometry_observed_native_crosscheck_pending`; the owner
automatically merges the exact pinned structural observation into the
same canonical `profiles/models/gemma/gemma-3/gemma-3-1b-it/profile.json`
using an atomic Git SHA check. Raw tokenizer values, tensors, local
paths and GPU UUIDs are not uploaded. `--no-git-sync` suppresses
publication without suppressing inspection.

The first on-host header attempt stopped at the earlier overstrict
`general.file_type == 24` assertion. That failed **before publication**,
so no model integrity failure should be inferred. Commit
`1a29ee2` changes the parser to compare the declared value with actual
matrix tensor types. Only the local reread can tell us what that pinned
GGUF actually declares; do not guess its value from the failure.

The native **Gemma3 context** remains intentionally blocked until these
GGUF hparams are reconciled against the pinned llama.cpp Gemma3 source,
especially sliding-window pattern, head count, K/V dimensions and the
minimum context allocation; even a successful header parse is NOT
enough to certify KV dimensions or GPU context fit.

### 5b — Compare pinned Gemma 3 GGUF hparams with llama.cpp source (READY; no CUDA)

The on-host, SHA256-verified header inspection was published into the
canonical model profile (commit `1ebed19`). Recorded source facts:
26 Transformer blocks, 4 attention heads, 1 KV head, K/V dimension
256 each, SWA window 512, GGUF context limit 32768, 183 BF16 and
157 F32 tensor descriptors. The BF16 matrix-tensor count dominates;
the advisory `general.file_type=32` is not defined in the pinned
ggml ftype enumeration (`MOSTLY_BF16=24`). The raw value and its
warning are retained rather than reinterpreted.

The pinned backend source `src/models/gemma3.cpp` invokes
`load_swa_pattern(ml, 6)` when the SWA window exists.
`src/llama-hparams.cpp` uses `il % 6 < 5` for SWA when the GGUF
does not override `gemma3.attention.sliding_window_pattern`.
No override appeared in this GGUF. Thus, for 26 blocks:

- SWA indices: all layers except 5, 11, 17, 23 (22 layers).
- Global indices: 5, 11, 17, 23 (4 layers).

This is **source-derived**, not GPU-allocated proof.
The canonical profile keeps the separate `gguf_header_observation`
and `source_derived_geometry` with pinned source revision and layer
indices; future host inspection enriches that same profile automatically.

With F16/F16 KV, SWA FULL ON, Flash OFF and `n_ctx=256`, a single
layer has 256 cells × 1 KV head × (256 K + 256 V) × 2 bytes = 262144
logical bytes = 0.25 MiB. Total is 6.5 MiB, split as 5.5 MiB SWA
and 1.0 MiB global. These are **logical bytes** and are not reported
CUDA allocations. The actual native cache and compute buffers remain
unknown pending the next guarded context measurement.

Use the source-only planner (no CUDA, no download or native weight load):

```bash
git pull --ff-only origin model-profiling

./ruth models kv-plan --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --quantization BF16 \
  --n-ctx 256 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn off --type-k F16 --type-v F16
```

Expected `status` is
`gemma3_kv_pinned_source_derived_native_context_unverified`.
It explicitly refuses to claim GPU fit or to authorize context loading.
Do not confuse `context_length=32768` (GGUF training/runtime metadata)
with measured CUDA fit on this host.

### 5c — Separate attention WEIGHT formats from KV activation precision

GGUF header metadata and geometry alone cannot determine the
storage type of every attention matrix. The inspector now captures
**tensor role, layer index, data type, and dimensions** for allowlisted
`blk.N.attn_q/k/v/output.weight` and attention-normalization weights.
It returns `attention_weight_tensors` and completeness/dtype counters
for each of Q, K, V and Output. Nothing reads actual tensor weight bytes
or initializes CUDA; these are GGUF tensor **descriptors** only.

Repeat the exact pinned local `inspect-gguf` command after `git pull`
to enrich the existing one-and-only canonical Gemma 3 profile. The
canonical Git uploader checks the previous header values are unchanged,
preserves native weight-allocation measurements and appends the new
per-layer inventory. It also stores a separate
`architecture.kv_precision_assessment_256`, without selecting a
new cache type automatically.

**Do not equate a BF16 K or V projection weight with a BF16 KV cache.**
The KV cache stores activations generated by the attention computation.
At 256 cells, 26 layers × 1 KV head × (256 K + 256 V):
- F16 K/V: 6.50 MiB logical
- BF16 K/V: 6.50 MiB logical
- F32 K/V: 13.00 MiB logical

The pinned `llama.cpp` source defines `GGML_TYPE_F16`,
`GGML_TYPE_BF16`, `GGML_TYPE_F32`. The native context accepts
separate `type_k` / `type_v` and forbids quantized V cache without
Flash Attention; it does **not** globally prohibit floating BF16 or
F32 Value cache. This establishes source-level possibilities, **not**
a proven CUDA kernel path, allocation or numerical behavior for this
host and GGUF. Ruth's guarded load-phase baseline remains **F16/F16**,
Flash OFF, SWA FULL ON until a separate approved native comparison.

BF16 has the same 2-byte storage as F16 and wider exponent range, but
fewer significand bits; F32 doubles cache storage and cannot restore
precision already lost in the original BF16 weights. There is no BF32
GGUF storage type in this pinned ggml revision (TF32 is different).

The `models kv-plan` response now includes
`kv_precision_candidates` with logical memory, attention-weight
provenance, and `automatic_cache_dtype_alignment_approved=false`.
That choice remains a runtime policy decision requiring explicit load
and, later, inference validation. Neither native load nor data
generation happens during header inspection or planning.

### 5d — KV storage precision planning, read-only (no GPU)

The published GGUF already records the attention tensor dtypes per layer.
`models kv-plan` now accepts F16, BF16 and F32 for Key and Value
independently, but **never approves a native context**:

```bash
./ruth models kv-plan --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --quantization BF16 \
  --n-ctx 256 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn off --type-k BF16 --type-v BF16

./ruth models kv-plan --purpose production --family gemma \
  --model gemma-3 --version gemma-3-1b-it --quantization BF16 \
  --n-ctx 256 --type-k F32 --type-v BF16
```

At 256 cells, logical totals are 6.5 MiB for BF16/BF16,
13.0 MiB for F32/F32, or 9.75 MiB for F32/BF16.
The cache dtype does **not** follow automatically from the GGUF attention
weight dtype. This is only a byte-size estimate: not an assertion
of native kernel support, allocation, GPU fit, numeric fidelity or
production eligibility. F16/F16 remains the designated baseline.
The separate `models probe-load` Gemma3 context gate stays closed;
no GPU load or inference is performed by `kv-plan`.

### 6 — Minimal native context allocation (EXPERIMENTAL; permission required)

Phase 26 adds a *separately authorized*, allocation-only probe for the
pinned Gemma3 BF16 model. Its exact allowlist is n_ctx=256,
n_batch=n_ubatch=64, SWA FULL ON, no CUDA preinit, with **either**
F16/F16 Flash OFF (baseline) **or** BF16/BF16 Flash ON (candidate).
Both require an explicit `--allow-gpu-load` and leave inference disabled.
Run the tests first; the step-by-step command and evidence handling are
documented in `docs/phase-26-gemma3-native-context-probes.md`.
No source-only result authorizes production inference, auto selection,
longer context or arbitrary cache types.

### 7 — Profile reuse and production eligibility

A successful context allocation is **not** production inference
qualification. Ruth may reuse exact-matching hardware/backend measurements
for load-fit planning, but new host CUDA/driver/kernel combinations must
be marked unknown or conservatively rejected until measured.
Automatic Git updates change the canonical profile, not create another
profiles/measurements/<digest>.json.

## Supported smoke-proof statements

- Step 1 can certify only source artifact metadata, not GGUF content beyond
  its published metadata.
- Step 2 can certify exact downloaded GGUF bytes and archive identity.
- Step 4 can certify weights allocation with full GPU offload when observed.
- Step 6 can certify a specific context allocation, not token correctness.
- No stage certifies semantic quality of an unquantized 1B model versus
  a quantized 9B model. A larger quantized model may outperform a smaller
  unquantized model: user priority is policy, not an objective comparison.
