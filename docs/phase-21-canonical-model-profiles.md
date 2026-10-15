# Phase 21 — One canonical model profile, floating formats, loading-only tuning

## Core decisions

One model identity is `family/model/version`, independent of
`--purpose production|test`. Variants (GGUF precisions and quantizations),
backend-specific rules and hardware/configuration observations are internal
sections of this one model profile.

The canonical Git profile is:

```
profiles/models/gemma/gemma-2/gemma-2-9B-it/profile.json
profiles/models/llama/llama-3.2/llama-3.2-1B-instruct/profile.json
```

The previous `profiles/measurements/<GGUF SHA>/<digest>.json` files remain
read-only provenance archives; new owner publications merge into the fixed
`profiles/models/.../profile.json` file. Every update fetches the current
GitHub blob, merges independently verified observations, PUTs with its
`sha` (optimistic compare-and-swap) and retries merge on concurrency.
Never delete an unrelated quantization, CUDA version or host observation.
An identical native measurement is not duplicated.

The owner upload path passes the **existing strict pinned-model evidence
validator** before Git mutation. Non-owners cannot silently write into
the owner's repository; external proposals require the historical explicit
PR permission/review workflow. GPU UUIDs, PCI bus addresses, host paths,
raw CUDA stderr and prompts remain only in private local evidence.

### Provenance tiers

- Source-verified: model architecture and GGUF metadata, when checked.
- Backend-version source-derived: 256-cell context padding and SWA rules.
- Estimated: 256-aligned K/V tensor bytes, never labeled as VRAM.
- Native-measured: actual weights buffers, K/V logical tensors and allocated
  CUDA buffers; record backend version and exact configuration.
- Host-verified fit: strictly corresponding driver/CUDA/architecture/binding
  and complete context options match.
- Inference-verified/production-qualified: **not inferred** from allocation.

UNKNOWN means `null`; missing proofs stay unknown. The host profile
matcher does not generalize a single run to a different CUDA backend,
driver, GGUF hash, Flash mode or KV precision. If a profile doesn't
exist, model selection constructs a model skeleton with missing values.
No external network or CUDA action occurs during skeleton creation.
The skeleton becomes persistent on Git through the next validated,
authorized owner measurement or explicit publication.

## Floating-point GGUF formats

Recognize `F32`, `F16`, `BF16`, plus lowercase GGUF filename variants
`f32`, `f16`, `bf16`. These are **weight storage types**, distinct
from `type_k/type_v` (KV cache tensor precision).

- F32: IEEE-like 32-bit floating, 4 bytes/weight (not all GGUF tensors
  need be this dtype).
- F16: 16-bit floating, 2 bytes/weight, shorter exponent range.
- BF16: bfloat16, 2 bytes/weight, broader exponent range than F16 but
  fewer significand bits.
- BF16 and F16 are not interchangeable, nor does one unconditionally
  preserve more quality. Verify original checkpoint dtype and weight
  conversion provenance. BF16 converted to F32 has twice the bytes
  but **cannot restore** precision previously lost.
- Quality ranks in quantization_policy.json are **heuristics** and
  valid only within the same model. Across different models, compare
  benchmark quality separately, not just F32 > BF16 > F16 or size.

New potential original-format production models are recorded as
unapproved, unpinned candidates in `catalog/unquantized_model_candidates.json`:
Qwen3-1.7B BF16 source or F16 GGUF, and Qwen3-0.6B BF16/F16 plus F32
for backend-format testing. Do not automatically download, select or
mark qualified candidates lacking immutable source revision, exact
GGUF SHA256, size and native probe capability. Source repositories:
https://huggingface.co/Qwen/Qwen3-1.7B,
https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF,
https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF and
https://huggingface.co/prithivMLmods/Qwen3-0.6B-GGUF.

## Loading-only baseline and batch characterization

For the pinned unmodified llama.cpp revision, **all NEW context load
probes request n_ctx = 256** (not 128). Extensions use 512, 768, 1024,
... in fixed 256-cell steps. The CLI enforces the multiples before
starting the native GPU worker. Older `n_ctx_requested=128` results
remain valid archival measurements but are not falsely relabeled as
canonical `n_ctx_requested=256`.

For this phase, lock:
```
GGUF = Gemma2 9B IT Q4_K_M
type_k = F16 ; type_v = F16
flash_attention = off
swa_full = true
n_seq_max = 1 ; kv_unified = false
n_ctx_requested = 256 ; n_ctx_effective = 256
```

Characterize `n_batch/n_ubatch` with the following **unmeasured
candidates**, evaluated one at a time after GPU preflight:

| Experiment | n_ctx | n_batch | n_ubatch | Purpose |
|---|---:|---:|---:|---|
| A | 256 | 64 | 64 | Canonical fixed-context baseline |
| B | 256 | 128 | 64 | Isolate logical batch |
| C | 256 | 256 | 64 | Full-context logical batch |
| D | 256 | 256 | 32 | Smaller compute microbatch |
| E | 256 | 256 | 128 | Larger microbatch, only if safe |
| F | 256 | 256 | 256 | Only with measured GPU headroom |

Each run records CUDA weights, global/SWA K/V, compute, free-memory
snapshots, worker success/failure and exact options. Do not automatically
run B–F on an unqualified 6GiB GPU: first compare A/C and inspect margins.

Batch decisions for *load-only* can minimize observed allocations and
instability but **cannot prove the best inference speed or quality**;
future decode benchmarks will determine throughput separately.

The known historical F16/F16 baseline requested `n_ctx=128`
(effective 256), `n_batch=n_ubatch=64`, weights CUDA 5488.4 MiB,
K/V 84 MiB, compute 63.38 MiB, CUDA free after context 44.81 MiB,
Flash OFF/ON both allocation-successful. These observations are
preserved, but a **new** requested-256 load baseline is still pending.

## Interfaces and limitations

```bash
# Read-only: same model profile for production AND test, inspect host if available
./ruth models select --purpose production --quantization Q4_K_M
./ruth models select --purpose test --family gemma --model gemma-2 \
    --version gemma-2-9B-it --format gguf --quantization Q4_K_M

# New fixed 256-cell context experiment: only execute on user host
./ruth models probe-load --quantization Q4_K_M --load-mode context \
  --n-ctx 256 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn off --type-k F16 --type-v F16 \
  --allow-gpu-load
```

The `model_profile_v2` resolver and evaluator use the new canonical
file. Owner sync merges there using `canonical_profile_git.py`.
The older `production_selection.py` and legacy memory profile matrix
still retain historical selection heuristics and inference qualification
gates; their complete unification into a universal multi-model
production orchestrator is not yet implemented. In particular, new
unquantized Qwen candidates are RESEARCH ONLY until formally pinned
and assigned backend-specific load support.

Nothing in this phase patches llama.cpp, downloads model weights,
executes a CUDA load in CI or silently confirms that a larger context fits.
