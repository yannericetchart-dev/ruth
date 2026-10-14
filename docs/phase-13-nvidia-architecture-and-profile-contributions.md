# Phase 13 — NVIDIA-only Ruth architecture and shared CUDA profiles

This policy intentionally **limits Ruth production hardware selection** to
NVIDIA Ada Lovelace, Hopper and Blackwell, with Tensor Cores generation >=4,
the CUDA compute capability observed on the **actual host**, and a matching
architecture-specific nominal VRAM tier. The nominal VRAM tier tolerance is
3% for Nvidia-SMI versus advertised GB/GiB values, to allow the 6141 MiB
RTX 4050 to match the 6 GiB Ada class. A memory tier is a **policy class**,
not proof that any particular NVIDIA SKU exists with that exact size.

| Architecture | Compute capability | Tensor Cores | First native CUDA | Ruth VRAM tiers (GiB) |
|---|---|---:|---:|---|
| Ada | 8.9 | 4 | 11.8 | 6, 8, 12, 16, 24, 48 |
| Hopper | 9.0 | 4 | 11.8 | 80, 94, 141 |
| Blackwell Data Center | 10.0, 10.3 | 5 | **12.8** | 180, 192, 372 |
| Blackwell Consumer/WS | 12.0 | 5 | 12.8 | 12, 16, 24, 32, 48, 96 |

Sources: https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html
and https://developer.nvidia.com/cuda/gpus

Blackwell DC **does not** have native CUDA 12.0 support in this matrix.
The Transformer Engine table is
https://nvidia.github.io/TransformerEngine/support_matrix.html :
FP8 modes require appropriate library/CUDA versions; NVFP4 is
restricted on sm_120 versus sm_10x; universal FP6 Transformer Engine support
must **not** be assumed. Architectural hardware precision is NOT proof that
llama.cpp/GGUF is using that Tensor Core format. Ruth records native
architecture compatibility as unverified until the CUDA binary genuinely
loads and executes the corresponding kernel.

## Separate CUDA version gates

Never conflate:
- `nvidia-smi CUDA Version` = **driver's maximum supported CUDA API**.
- `nvcc --version` = **installed build Toolkit** (may be absent).
- `cu132` = CUDA target of Ruth's currently pinned native backend wheel/build.
- `Compute Capability` = native GPU ISA / architecture.
- `Latest reviewed CUDA` = versioned advisory policy (currently 13.4),
  not an automatic network latest check or forced upgrade.

Ruth rejects verified versions below the **architecture minimum**, reports
unverified missing data as unknown, and warns about older-but-compatible
Toolkits/backends/drivers. A driver reporting a **lower CUDA major** than the Toolkit or chosen
backend is incompatible. When only the **minor** version differs within a
major CUDA family and the driver meets NVIDIA's family minimum, Ruth warns
about restricted features/PTX and requires runtime validation instead of
rejecting all minor-version-compatible software. The existing
`setup` and pinned wheel remain CUDA **13.2** for now; supporting another
CUDA version and GPU architecture requires rebuilding/verifying that native
backend. Being an allowed hardware class does **not** certify current
`llama-cpp-python` kernels for Hopper/Blackwell.

The baseline release mapping and policy live in
`catalog/nvidia_architecture_policy.json`. It may be updated after review,
not by inferring from whatever `nvidia-smi` prints on an unknown machine.

## Commands

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling
.venv/bin/python -m unittest \
  tests.test_phase12_model_memory_profiles \
  tests.test_phase13_hardware_architecture \
  tests.test_phase14_profile_registry \
  tests.test_phase15_profile_contribution -v

# Check actual GPU compute capability, Tensor Core generation,
# strict arch/VRAM class, driver maximum CUDA, optional nvcc Toolkit.
./ruth models hardware

# Host eligibility + available, unverified and unsupported model profiles
./ruth models profiles --host

# First time model source is verified via allowed provider metadata.
# This creates a local empty profile: unknown native measurements remain null.
./ruth models profiles --initialize \
  --quantization Q4_K_L --allow-metadata-network

# After a downloaded, SHA-verified GGUF has been probed (no inference):
./ruth models probe-load --load-mode weights --cuda-preinit \
  --allow-gpu-load > /tmp/ruth-weights.json

# The OOM exit code is expected; its JSON remains valid evidence.
# A fresh probe is needed: older reports predate model_verified/source fields.
./ruth models profiles --host --quantization Q4_K_L \
  --evidence-file /tmp/ruth-weights.json --record-local

# Owner automatically syncs on --record-local, if 'gh' is authenticated.
# External contributors can explicitly open their review PR:
./ruth models profiles --profile-file PATH_TO_SAVED_PROFILE --sync-git
```

No GitHub credentials must be pasted into messages or committed to Ruth.
Ruth checks the authenticated GitHub identity. For the repository owner,
the privacy-reduced evidence is committed **directly** to `model-profiling`
in `profiles/measurements/`, without a fork or a PR. Other users can only
submit via `profiles/submissions/` and a review PR. All contributions
are schema-checked; the repository must additionally enforce owner review
through branch protection / CODEOWNERS. If GitHub access is missing, local
measurement records are retained and synchronization can be retried.
Unauthenticated clients cannot write to the central Git repository.

## Reusable profile separation

1. `catalog/nvidia_architecture_policy.json`: reusable architecture,
   capability, Tensor Core, permitted VRAM, CUDA minimum and TE format policy.
2. `catalog/model_catalog.json`: pinned GGUF source/revision/SHA/size/license
   (never store a device-specific OOM in this catalog).
3. `.ruth-build/profiles/models/<gguf_sha256>.json`: on-host persisted
   evidence. First created after explicit verified source metadata, with
   `native_cuda_weights_mib=null`, `runtime_measurements={}` and
   `context_profiles={}`; source verification does **not** accept the license,
   download or load the GGUF.
4. The runtime key separates `llama.cpp`, GPU architecture, exact backend
   CUDA version, driver-reported CUDA max, Toolkit version (or unknown), exact
   driver version and pinned Python native binding version. Memory profiles
   from one CUDA version **must not** be silently reused as measurements
   from another version.
5. Each context key must also include `n_ctx`, Flash Attention mode,
   K/V dtype, and backend-specific KV padding policy. Unsupported and
   unmeasured entries remain null; no computed fake KV sizes.
6. Public `profiles/submissions/` entries are untrusted proposal **samples**,
   not authoritative compatibility declarations. Merge/reconciliation
   into curated profiles is a separate review step.

The runtime record tracks driver reserved VRAM, observed CUDA initialization
NVML-to-CUDA free-memory delta, GPU weights buffer and (when measured)
native KV and compute buffers. Subsequent observations exceeding the prior
same-version maximum emit a warning and update the maximum. Changes due to
different GPU models, contexts, CUDA versions or kernels cannot be attributed
to a single scalar; never invent a magic reserve percentage.

## Context/KV padding: not inferred from weights OOM

At `Q4_K_L` on the RTX 4050, the model weights fail before any KV allocation.
Context/padding/Flash Attention evidence is still **unknown**. The old
llama.cpp cache padding may use 32 cells when FA off or 256 when FA on,
but graph-effective KV padding and physical cache layout differ and can
vary with runtime version and backend. Accurate profiling must record
native `kv_size`, allocated K/V and compute MiB, actual FA mode,
K/V dtype and context size across 128,255,256,257,511,512,513.
The current worker does not prove native FA mode or KV dtype, so records
those properties as null even when context allocation succeeds.

## Outstanding gates

- The project-wide legacy `doctor/setup` still has the pinned 13.2/R595
  installation policy; the new hardware checker is not a blanket replacement
  for build/runtime ABI verification. Do not claim operational multi-arch
  support until per-architecture wheel/kernel tests succeed.
- Auto-merge **must not** trust contributor-submitted numbers. Github PR
  validation only checks provenance/schema, not physical authenticity.
- No hosted central service or unattended installed GitHub App exists yet;
  for unattended periodic publication a user-controlled GitHub App
  installation/token strategy and repository-maintainer approvals are needed.
- Public profile submissions intentionally omit PCI BusID, GPU UUID,
  process lists, user paths, prompts and credentials.


## Publication retry when `gh` is missing

A `--record-local --publish-pr` command first atomically saves the
measurement under `.ruth-build/profiles/models/<GGUF-SHA256>.json`.
Publication can then fail independently, for example if `gh` is absent.
**Do not rerun the native CUDA probe just to retry Git publication.**
The profile registry also deduplicates identical report JSON, avoiding
duplicate observations even if it is recorded again.

After installing GitHub CLI on Debian/Ubuntu via the approved distribution
package (`sudo apt update && sudo apt install gh`), authenticate with
`gh auth login` and confirm `gh auth status`. Then locate the already
saved profile and publish that exact file:

```bash
find .ruth-build/profiles/models -maxdepth 1 -type f -name '*.json' -print
./ruth models profiles --sync-git \
  --profile-file .ruth-build/profiles/models/<GGUF-SHA256>.json
```

In the code, `git_publication.status=git_cli_missing` is now a **retryable
publication issue**, reported separately from the successful
`local_profile_update`. An explicit `--sync-git` returns nonzero if neither an owner commit
nor a contributor PR is confirmed; the saved profile is preserved. Local listing and native memory
measurements are unaffected.

Users with a different GitHub login submit a PR from their fork,
requiring the repository owner's review. The repository owner commits
directly to `model-profiling` without a branch or PR proposal.


## Git sync: trusted repository owner versus external users

Ruth now authenticates the account via `gh api user`, **never by an arbitrary
username in a local file or a provided command-line argument**.

- Authenticated `yannericetchart-dev`: when `--record-local` or
  `--initialize` creates/updates a local profile, Ruth **automatically
  commits the privacy-reduced measurement directly** to the
  `yannericetchart-dev/ruth` **`model-profiling`** branch. No fork, PR, or
  per-measurement authorization is needed. `--no-git-sync` opts out for one
  invocation. `--sync-git --profile-file PATH` retries a previously saved
  measurement without recording it a second time. Existing immutable files
  are detected as already current instead of overwritten.
- Other authenticated contributors: Ruth saves data locally but does not
  upload it implicitly. When they explicitly request `--sync-git`, a fork PR
  targets `model-profiling`. **The repository owner must approve/merge**
  that contribution. Having collaborator permissions is not sufficient to
  take the owner-direct route.
- `gh` missing, unauthenticated, insufficient GitHub rights, rejected branch
  protection, network errors: **local profile retained**; the diagnostic
  exposes `git_publication.status` separately and does not claim that a
  remote Git update succeeded.

Owner-direct public profiles live in the **append-only**
`profiles/measurements/<GGUF-SHA256>/<content-digest>.json` directory on the
branch. Contributor proposals use
`profiles/submissions/<GGUF-SHA256>/<content-digest>.json` until reviewed.
These files contain only vetted fields, not machine UUID, PCI BusID, local
paths or prompts. Both paths undergo schema/integrity validation in CI. A
commit directly to `model-profiling` is still **not** evidence of successful
inference when only loading-memory measurements are available.

The direct route uses GitHub's Contents API, and GitHub enforces the actual
authenticated identity and branch permissions. The repository owner must
already have `gh` installed and authenticated once; no Ruth-specific
publication token or recurring confirmation prompt is required.

### Retry the existing local Gemma profile, without a new CUDA probe

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling
gh auth status
PROFILE=".ruth-build/profiles/models/$(.venv/bin/python -c 'import json; print(json.load(open("/tmp/ruth-weights.json"))["model_sha256"])').json"
test -f "$PROFILE"
./ruth models profiles --profile-file "$PROFILE" --sync-git
```

The legacy `--publish-pr` option remains an alias to explicit sync; for
the owner it now commits **directly**, not through a PR.
