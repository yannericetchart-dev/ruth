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

### 5 — Native GGUF hparams / context geometry (NOT APPROVED YET)

Add exact-source/model-specific metadata and context allocation checks
after observing the native Gemma3 GGUF tensor/layout/hparams. Explicitly
verify n_layer, n_head_kv, key/value head sizes, SWA pattern/window and
backend applicability. Do not pass Gemma 2 geometry into the Gemma 3
KV planner.

### 6 — Minimal canonical context loading (NOT APPROVED YET)

Use `n_ctx=256` and profile `n_batch=64,n_ubatch=64` first, KV F16/F16,
Flash OFF, SWA FULL ON. Only after validated hparams and staged
authorization should this be enabled for Gemma3. Probe no inference
tokens. Tune n_batch/n_ubatch later in discrete, memory-guarded tests.

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
