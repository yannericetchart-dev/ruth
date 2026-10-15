# Phase 18 — llama.cpp KV/SWA profiling and experimental padding proposal

## Scope: original upstream native backend ONLY

Ruth production keeps the **unchanged** llama.cpp commit
`0c1e57098bba43ac29e6e3b677cdceebdd22334f`,
llama-cpp-python `0.3.36`, and the exact existing CUDA native build.
**No source patch, custom wheel, or automatic CUDA experiment is introduced in
this phase.** Backend-specific rules do not apply to other engines or
llama.cpp revisions without a newly reviewed version policy.

The pinned rules live in `catalog/llama_kv_backend_policy.json`;
read-only estimates are in `ruth_runtime/llama_kv_profile.py`.
`./ruth models kv-plan` never initializes the GPU or loads a GGUF.

## Data model

The backend policy tracks three independent values from pinned upstream code:

- **Context alignment 256 cells:** `src/llama-context.cpp:L289-L305`.
  This rule is **independent of Flash Attention ON/OFF/AUTO**.
- **SWA cache alignment 256 cells:** `src/llama-kv-cache-iswa.cpp:L69-L105`.
- **Minimum graph-view KV alignment 256 cells:**
  `src/llama-kv-cache.cpp:L1260-L1273`. This graph/view extent is NOT
  a separate resident K/V allocation; do not double-count it.

The model geometry is different from backend policy: Gemma 2 9B's
42 attention layers, 21 global layers, 21 SWA layers, 4096-token SWA window,
8 KV heads of 256 K/V elements per head, and layer assignment pattern.
These are **source model defaults requiring cross-check against the actual
GGUF/native layer metadata**. A future model architecture may have unequal
K/V head sizes or a per-layer pattern; do not transplant Gemma defaults.

For the current audited single-sequence, non-unified context path:

```text
C = PAD(n_ctx_requested, 256)
B = min(n_ctx_requested, n_batch_requested)       [causal model]
U = min(B, n_ubatch_requested)
S_ON = C
S_OFF = PAD(min(C, SWA_WINDOW + U), 256)
Global KV cells/layer = C
SWA KV cells/layer = S_ON or S_OFF
```

If `n_seq_max > 1` or `kv_unified=true`, the current calculator fails
closed instead of silently applying single-sequence rules.
The precise effect of multi-sequence stream count must be characterized
and validated separately.

For F16 K/V and current Gemma 2 9B at `n_ctx=128`, 256 internally
allocated cells explain **84.00 MiB of logical K+V tensors**. These
logical tensor bytes are not the CUDA allocation size and exclude compute
and allocator overhead. The first SWA saving for `n_ubatch=64` occurs
at `n_ctx_effective=4608` (4352 SWA cells), **42 MiB logical savings**
relative to `swa_full=true` at the same context.
This threshold changes with `n_ubatch` and the 256-cell padding.

## Batches, Flash Attention and mixed K/V types

Record `n_ctx_requested`, `n_ctx_effective`, `n_batch_requested`,
`n_batch_effective`, `n_ubatch_requested`,
`n_ubatch_effective`, `swa_full`, and both
`flash_attention_requested` and `flash_attention_effective`.
A requested `AUTO` is **not** evidence of effective ON:
the native backend may disable unsupported fused operations.

Record **K and V independently**. The independent ggml types currently
allowed for diagnostic selection are F16, Q8_0, and Q4_0; byte formulas
use the exact ggml block size and bytes/block for the pinned library.
`type_v` quantized requires Flash Attention (AUTO or ON); `OFF`
must reject before native GPU allocation. Other formats are unknown
until backend capability validation.

`n_batch` caps the logical prompt batch; `n_ubatch` controls physical
microbatch graph sizing, available memory and SWA cache capacity.
Compare performance, peak GPU usage, and allocation stability separately.
Changing these values is **not** a universal win: smaller microbatches
can reduce compute buffers at the cost of latency and throughput.

The native direct-C load probe now receives explicitly selected options
in a strictly allowlisted isolated-worker environment config; the
read-only parent validates them before GPU activity. Its native stderr
parser extracts **global vs SWA**, group cell counts, K/V logical tensor
sizes, backend CUDA buffer sizes and effective context/batch settings.
Missing logs remain `null`. Physical K/V byte split remains unknown
if backend allocation combines tensors in one buffer.
Native-logger effective Flash Attention stays null unless genuinely
observed, especially for AUTO. Historical context measurements remain
legacy observations, not retroactively upgraded to these new fields.

New profile keys include backend commit, Flash mode, K type, V type,
SWA, n_batch, n_ubatch and context size. Git publication validates a
source-derived checksum-independent calculation against the pinned
backend policy and prevents mislabeling source calculations as native
measurements or verified production inference.

## Commands

```bash
# No model, no GPU/CUDA; source-based estimates only
./ruth models kv-plan --purpose production --n-ctx 128 \
  --n-batch 64 --n-ubatch 64 --swa-full on \
  --flash-attn off --type-k F16 --type-v F16

# Prediction of SWA memory savings at long context (no GPU)
./ruth models kv-plan --n-ctx 4608 --n-batch 256 \
  --n-ubatch 64 --swa-full off --flash-attn on \
  --type-k Q8_0 --type-v Q8_0

# Optional, separate, explicitly authorized native CUDA context
# experiment (only if offline host-memory preflight says feasible).
./ruth models probe-load --quantization Q4_K_M --load-mode context \
  --n-ctx 128 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn off \
  --type-k F16 --type-v F16 --allow-gpu-load

# The probe now AUTOMATICALLY archives its JSON evidence under
# .ruth-build/profiles/evidence/, updates the model profile, and attempts
# owner-only Git synchronization after measurements. No second CUDA run
# or manual JSON copy is necessary for NEW probes.

# To keep evidence local and explicitly avoid Git synchronization:
./ruth models probe-load --quantization Q4_K_M --load-mode context \
  --n-ctx 128 --n-batch 64 --n-ubatch 64 \
  --swa-full on --flash-attn off --type-k F16 --type-v F16 \
  --allow-gpu-load --no-git-sync

# For an ALREADY completed probe from before auto-record was added, one
# manual evidence import may still be required; importing does NOT load CUDA:
./ruth models profiles --host --record-local \
  --evidence-file /path/to/previously-saved-probe-output.json

```

## Probe recording lifecycle

The `probe-load` command previously returned
`runtime_profile_published=false` and left recording to
`models profiles --record-local --evidence-file ...`. This was an
orchestration gap. **New successful native weights/context probes (and native
failures with actual measured CUDA buffers) now record automatically.**

1. The raw JSON report is archived locally under
   `.ruth-build/profiles/evidence/<gguf-sha>-<digest>.json`, with an
   immutable canonical digest. The local copy retains hardware identity
   solely for strict evidence validation.
2. The local GGUF profile is atomically updated; exact repeated evidence
   is idempotent.
3. A strictly sanitized proposal removes UUID, PCI bus, local paths and
   raw stderr, and an authenticated repo owner synchronizes directly with
   `model-profiling`. Non-owners retain their observations locally,
   without a hidden fork or PR.
4. The output reports `profile_recording_status`,
   `local_profile_update`, `local_evidence_file`,
   `git_publication` and `runtime_profile_published`. Only verified
   publication makes the final field `true`; local recording does not.
5. A Git failure never discards the native report or the archived local
   evidence. Retry publication using `models profiles --sync-git
   --profile-file <local-model-profile-path>` once authentication is fixed.
6. `--no-git-sync` prevents publication while retaining local evidence.

The older probe run is **not retroactively captured**. If its original
JSON was not saved, Ruth cannot reconstruct it from the local CUDA worker.
Its user-supplied structured output can be imported once without running
CUDA, with the usual host/provenance checks.

The default **direct C** context probe still selects Flash Attention
`AUTO`, while the historical **Python inference** path used `OFF`.
They are different configurations. Do not compare their compute
allocations as if they were identical. Align both test paths explicitly
before any production qualification campaign.

## Future idea — patched llama.cpp padding 128 / 64 (DOCUMENTATION ONLY)

**Ruth has NOT patched, built, selected, validated or published a
modified llama.cpp native backend.** The existing wheel and all prior
profiles remain tied to the upstream commit above.

There is a credible potential memory benefit for a marginal 6 GiB GPU:
with full Gemma2 F16 KV, a 128-token context rounded to 256 cells uses
84 MiB of logical KV tensors. If a coordinated backend patch safely
allocated only 128 cells, the formula predicts 42 MiB, a possible
**42 MiB reduction** before any compute or allocator consequences.
This could be meaningful when the observed full-GPU inference margin
is only several tens of MiB.

However, changing only the line
`cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256)`
is not safe as a tuning implementation. The 256 alignment also appears
in `n_ctx_seq`, SWA cache sizing and the graph-facing
`get_n_kv()` (which uses a minimum of 256).
A context cache of 128 cells with SWA still padded to 256 and graph
views expecting 256 cells may cause inconsistent sizing, access bounds,
graph incompatibility or even *larger* SWA allocations.

A future controlled experiment should:

1. Keep the upstream native build intact and create a separate patch-set
   tied to exact `llama.cpp` source commit.
2. Review every interacting context/SWA/graph padding site and CUDA
   kernel constraint; test 128 first, 64 only if supported.
3. Build a **separate** wheel and record patch digest, source commit,
   native shared-library and wheel SHA256, CUDA toolkit and SM target.
4. Test adjacent boundaries (64, 65, 127, 128, 129, 255, 256, 257)
   across Flash ON/OFF, separate K/V formats, SWA ON/OFF, batches and
   architectures, including sanitizer/invalid-graph failures.
5. Compare correctness, token equality within allowed tolerances,
   graph reuse, prefill/decode throughput, GPU peak allocation,
   KV/compute bytes and host stability to the unchanged build.
6. Never promote results between patched and unpatched runtimes or
   infer an operationally safe context solely from a smaller KV tensor.

**Decision:** defer all native patching until exact-option profiling of
the unmodified pinned build is complete. The policy will refuse an
unrecognized 128/64 padding configuration, not silently accept it.
