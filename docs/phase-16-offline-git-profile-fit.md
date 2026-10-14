# Phase 16 — Offline Git-based GPU model compatibility diagnosis

## Verified Git source

The `model-profiling` branch contains an owner-published measurement for:

- Gemma 2 9B IT GGUF `Q4_K_L`, pinned SHA256
  `ed8573f0d1e3ca58e471498fd162a65d4f4253dffdec28a7ba394fdcae92f1f3`.
- `profiles/measurements/<sha>/a23b210a1be60bda031d2e294ca3aeba12c0c58ef05ce4f377c16c07b43e000c.json`.
- Ada `sm_89` (RTX 4050 test), driver `595.99.02`,
  CUDA backend `13.2`, llama-cpp-python `0.3.36`.
- Native GGUF CUDA weights buffer: **5700.31 MiB**.
- NVIDIA driver-reported reservation: **369.00 MiB**.
- NVML free to CUDA Driver API free delta observed after primary context
  initialization: **90.19 MiB** (a cross-API, cross-time observation, not
  exact CUDA context allocation).
- `n_ctx`, Flash Attention, CUDA KV cache and compute memory: **unknown**.

Git published data is **not** proof of the RTX 4050's available free VRAM
at this moment. A read-only offline explanation must use previously stored
measurements and report the historical basis.

## Zero GPU queries / zero model loads

Once the repository has been updated with the committed record and a local
per-GGUF profile is present (saved from the earlier `--record-local` step):

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling

./ruth models profiles --offline-fit --quantization Q4_K_L --vram-gib 6
```

No `nvidia-smi`, CUDA initialization, GGUF opening or inference takes
place. No network calls are made by the Ruth command; the preceding
`git pull` is the independent operation that refreshes Git-tracked data.

The command takes architecture, exact driver version, backend CUDA and
llama-cpp-python version from the **previously saved local profile**.
It requires an explicit `--vram-gib 6` for older local profiles, because
that physical capacity was not persisted in earlier registry versions.
A recent saved host snapshot can supply it without that option.

For hypothetical machines or a missing local runtime identity, provide
exact selectors as required:

```bash
./ruth models profiles --offline-fit --quantization Q4_K_L \
  --architecture ada --vram-gib 6 --driver-version 595.99.02 \
  --cuda-backend 13.2 --native-binding 0.3.36
```

Unknown or conflicting identities stay `offline_fit_unverified`.
Every Git JSON is individually checked against its immutable filename
SHA256, pinned catalog model source and versioned architecture policy
before using its measurements.

## Mathematical proof for the observed Ada/6 GiB configuration

Use the **nominal 6 GiB = 6144 MiB** class, which is slightly higher than
the actually observed `nvidia-smi` 6141 MiB and therefore is favorable
to a theoretical full-GPU load:

```text
VRAM theoretical ceiling                  6144.00 MiB
Driver reserve (published)                -369.00 MiB
NVML->CUDA free difference (published)     -90.19 MiB
----------------------------------------------
Estimated available for GGUF weights      5684.81 MiB
GGUF native CUDA weights buffer          -5700.31 MiB
----------------------------------------------
Deficit                                     -15.50 MiB
```

Therefore `full_gpu_weights_cannot_fit_nominal_bound` on this
**matching Ada/driver/CUDA/llama.cpp version**, without creating a
context/KV cache and without probing the GPU. The direct historical
measurements from the host gave **5682.81 MiB** CUDA free before the
load and **17.50 MiB** actual deficit. That exact CUDA free value is
not present in the first published Git measurement; it is known from the
earlier private probe and can be retained in newer **local-only**
`host_snapshot` records.

A positive difference on 8 GiB does **not** certify model runtime
compatibility or inference: it means only
`weights_possible_context_unverified`. If a different driver, CUDA
backend, model hash, or llama binding is requested and exact evidence
does not exist, return `offline_fit_unverified`, never transplant
a result as confirmed for that different configuration.

New local `--record-local` calls save a historical host memory
snapshot with observed `memory.total`, `memory.free`,
`cuda_free_after_primary_ctx_init` and nominal VRAM class.
This snapshot is **not** uploaded to Git or considered live status.
Duplicate evidence records can update the snapshot without incrementing
observation count.

## Test coverage

```bash
.venv/bin/python -m unittest tests.test_phase16_offline_memory_fit -v
```

Checks include exact negative bounds for 6 GiB, unknown vs. possible
for 8 GiB, no CUDA/GPU subprocess or network calls, matching source
integrity, unknown runtime identities, and historical vs live memory
separation.
