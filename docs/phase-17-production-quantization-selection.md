# Phase 17 — Production Gemma GGUF quality-first selection

## Objective

When Ruth is asked for production model `gemma-2-9B-it`, choose the
highest-ranked *feasible* quantization from a **pinned approved GGUF source**.

The pipeline is a deterministic state machine, **not** an unconditional
download or inference action. An explicit `--execute` and the existing
model license, download, CUDA allocation and GPU inference authorization
flags are required for side effects.

## Decision tree

1. Read host GPU capability and **physical total** memory using read-only
   NVIDIA telemetry; reject unsupported architecture/CUDA/VRAM policy.
2. Retrieve GGUF variants at the repository's exact pinned Git revision
   (not mutable `main`). Validate source, license, LFS SHA256 and exact
   file byte size. Order by the versioned, heuristic quality policy.
3. If serialized GGUF bytes exceed the observed physical VRAM bytes,
   label `policy_excluded_gguf_file_larger_than_host_vram` and never probe
   the candidate. This is Ruth's conservative **file-size selection policy**,
   not a mathematical proof that CUDA weight buffers exceed VRAM.
   It is nevertheless a strong early pruning heuristic.
4. For pinned variants, read the exact model SHA / GPU architecture /
   CUDA backend / driver / llama-cpp-python measurement profile already
   committed under `profiles/measurements/` without initializing CUDA.
   A negative `full_gpu_weights_cannot_fit_nominal_bound` excludes that
   variant without rerunning it.
5. A historical positive **minimal-context GPU token** result is reused
   only at exactly the same GGUF, architecture, CUDA backend, driver and
   native binding version. A new `n_ctx` or missing KV/Flash Attention
   memory calibration is not production-qualified.
6. Choose the first still-unknown pinned quantization with a supported
   full-GPU probe. Download only that exact checksum-pinned GGUF (or reuse
   an existing verified local file); run native weights-only, then native
   n_ctx=128 context/KV/compute allocation, then isolated minimum-context
   token generation. All GPU operations require preflight.
7. Store native weights and context observations locally per GGUF SHA and
   runtime, including nulls for unobserved fields. If authenticated as the
   repository owner, submit sanitized immutable reports DIRECTLY to
   `model-profiling`. Other users only retain local observations until
   they separately authorize a fork PR for owner review.
8. An actual full-offload GPU minimal-context success is stored separately,
   then published with the memory report. Later requests can skip the same
   quantization's identical verified minimal-context test. This **does not**
   imply a production-context qualification.

### Expected Ada 6 GiB case

| Model | GGUF file | Selection result |
|---|---:|---|
| Q8, Q6, Q5_K_L/M/S | 6.48GB or larger for Q5_K_S and above | file-size policy rejection |
| Q4_K_L | 5.98 GB | measured 5700.31 MiB native weights > 5684.81 MiB estimated nominal budget, skip |
| Q4_K_M | 5.76 GB | first previously unverified pinned candidate, proceed |
| Q4_K_S and smaller | 5.48 GB and below | provisional capacity candidate, cannot claim tested |

Reference: https://huggingface.co/bartowski/gemma-2-9b-it-GGUF
Actual source inventory may not contain every theoretical quantization
listed in the shared ranking policy.

## Commands

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling

# Selection only: read-only host metadata + remote HF metadata;
# zero downloads and zero CUDA allocations.
./ruth models auto-production --allow-metadata-network

# AFTER reviewing the model owner's license, explicitly authorize
# downloads, weights/context GPU allocations and minimal token inference.
# A 5.76GB Q4_K_M download can occur here. Do not run without intending it.
./ruth models auto-production --allow-metadata-network \
  --execute --allow-model-download --accept-model-license \
  --allow-gpu-load --allow-gpu-inference

# Regression tests are simulated, no HF download / GPU needed.
.venv/bin/python -m unittest \
  tests.test_phase04_model_catalog \
  tests.test_phase05_candidate_selection \
  tests.test_phase06_model_download \
  tests.test_phase14_profile_registry \
  tests.test_phase15_profile_contribution \
  tests.test_phase16_offline_memory_fit \
  tests.test_phase17_auto_production -v
```

The plan lists **all actual discovered GGUF variants** in descending
quality with one of these states: file-size policy exclusion, measured
weights incompatibility, exact minimal-context success already in Git,
requires pin, requires backend support, or first candidate requiring
download and native tests. There is no fallback to CPU or lower quants
without fresh explicit review.

## Why smaller quantizations are NOT automatically PASS

A verified Q4_K_M model does **not** mathematically prove that Q4_K_S,
IQ4_XS, Q3, etc. are fully GPU/inference compatible: different tensor
formats, CUDA kernel support, KV formats and compute graphs can fail for
independent reasons. They are **size-filter-pass / lower-priority
candidates**, not `gpu_inference_verified=true`. Ruth avoids
unnecessary lower-quality tests after finding a verified higher-quality
candidate, but never fabricates validation for untested GGUF hashes.

Similarly, an 8 GiB nominal memory fit is not production qualification.

## Publication and future model-profile reuse

- Model pin: exact GGUF SHA256/byte length/revision, no unauthenticated
  source accepted.
- Weights: model CUDA allocation or diagnosed OOM, free/reserved CUDA
  historical telemetry tied to version.
- Context: `n_ctx_requested`, `kv_size_native`, K/V, compute buffer
  measured MiB, backend and unknown Flash Attention or KV dtype/padding.
- Minimal inference: exact `n_ctx`, generated token count, observed
  full CUDA layer offload and GPU process with exact runtime identity.
  A success at n_ctx=128 is **not** enough for operational production
  context qualification.
- Publication: owner automatically writes immutable allowlisted JSON to
  `profiles/measurements/<SHA>/<digest>.json` in `model-profiling`
  if GitHub CLI is available/authenticated. Others cannot write directly.
- If the same profile is already Git-published, the planner reuses it
  without redoing model load, context allocation or inference.

Unknown context/KV/Flash Attention parameters remain **null**, never
estimated as observations.
