# Phase 27 — Gemma 3 BF16: prefill/decode and numerical KV precision

## Experimental status and prerequisites

Phase 26 successfully initialized **all 18** F16/BF16/F32 K/V ×
Flash ON/OFF native CUDA contexts without running inference.
This removes an allocation-stage incompatibility hypothesis but
does **not** establish that all combinations can evaluate model tokens.

This next phase uses the *same* pinned Gemma3 GGUF, native CUDA wheel,
GPU and context geometry. All tests and commands use Ruth's project
`.venv/bin/python`. The benchmark does not alter the production policy.

- Pinned model: Gemma 3 1B IT BF16, SHA256
  `b144bde489651d3d292d7e47a59c6f811cdd96d184be582e5ad30ab338b9b6bf`.
- Pinned llama.cpp revision:
  `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- Exact pinned llama-cpp-python CUDA wheel SHA256:
  `ad5057352154d90f95a473d01d039c975f1987cd9b25f8ad7a83cf6c7e80ee74`.
- `n_ctx=256`, `n_batch=n_ubatch=64`, SWA FULL, one sequence,
  KQV GPU offload, Flash explicitly ON/OFF.
- No downloads, no model-format changes, no context escalation,
  no CPU fallback, no automatic cache dtype selection, no Git publication.

## Workload specification

- Fixed public test prompt; identical tokenizer/model for all 18 runs.
- One warm-up followed by **three measured repetitions per configuration**.
- One batched prefill, followed by **six separate decode calls**.
- Decode tokens are **teacher-forced and identical**, not sampled from the
  model. The native calls evaluate logits but do not generate chat text.
- Each call is synchronized with `llama_synchronize` before measuring
  elapsed time.
- All **seven** full-vocabulary float32 logits rows (prefill + six decode)
  from the final repetition are saved in a compressed, private, local
  `.npz` file under `.ruth-build/phase27/gemma-3-1b-it/`.
- `inference_executed=true` on success (model evaluation DID happen);
  `generated_tokens=0` means no free-form generation or sampling.
- Each case is isolated in a fresh worker process. Failure stops a
  sequential batch, without retrying with another dtype/Flash mode.
- Native/CUDA wheel integrity, pinned GGUF SHA, GPU identity,
  preflight occupancy, full GPU offload and return schema are enforced
  before accepting timing and numerical measurements.

The reported times are end-to-end, synchronized native wall-clock timing,
**not** device-kernel-only execution time; CPU overhead can be included.
Memory peaks from NVML are sampled and may miss short-lived allocations.

## Offline tests and read-only plan

```bash
cd ~/Dev/ruth-model-profiling
git fetch origin
git switch feat/gemma3-phase27-numeric-benchmark
git pull --ff-only

.venv/bin/python -m unittest \
  tests.test_phase26_gemma3_context_probe \
  tests.test_phase26_gemma3_context_matrix \
  tests.test_phase27_gemma3_benchmark -v

.venv/bin/python -m ruth_runtime.gemma3_benchmark
```

Without `--execute`, this only prints the plan and **does not access
GPU, load weights or run inference**.

## First authorized GPU inference benchmark — reference

This command runs actual inference, not a context-only probe. Its
explicit authorization is local to the command:

```bash
.venv/bin/python -m ruth_runtime.gemma3_benchmark \
  --case 1 --execute --allow-gpu-inference
```

Review the JSON result: `native_teacher_forced_inference_observed`,
`full_gpu_offload_observed=true`, logits artifact path and SHA256,
prefill and decode times. Failure ends the investigation until inspected.

For the next comparison, keep the same workload and test BF16 with
Flash OFF (Phase 26 matrix case 3):

```bash
.venv/bin/python -m ruth_runtime.gemma3_benchmark \
  --case 3 --execute --allow-gpu-inference
```

The Phase 26 allocation-successful BF16/BF16 Flash ON configuration is
case 2 and can be benchmarked separately after case 1:

```bash
.venv/bin/python -m ruth_runtime.gemma3_benchmark \
  --case 2 --execute --allow-gpu-inference
```

Only after reviewing the initial run outcomes, consider the remaining
cases one at a time with `--case N`. To run a sequential subset with
automatic stop on failure, use `--start-at N --execute
--allow-gpu-inference`. Unlike Phase 26, this launches **real GPU model
evaluation** for every selected case, which requires the explicit flag.

## Offline full-logit comparison

Take `logits_artifact.relative_path` from two previously successful
runs of the benchmark and run:

```bash
.venv/bin/python -m ruth_runtime.gemma3_logits_compare \
  --baseline '.ruth-build/phase27/gemma-3-1b-it/flash-off_k-F16_v-F16_EXAMPLE.npz' \
  --candidate '.ruth-build/phase27/gemma-3-1b-it/flash-off_k-BF16_v-BF16_EXAMPLE.npz'
```

Replace the `EXAMPLE` file names with paths from the actual JSON
reports. No GPU is used by this comparison. It requires identical
vocabulary size, prefill and teacher-forced decode token IDs.

Metrics:
- Maximum, mean absolute and RMS logit differences (float64 computation).
- Top-1 token agreement fraction.
- Forward/reverse KL divergences of full-vocabulary softmax.
- Jensen–Shannon divergence and per-position differences.

F16/F16 Flash OFF is a **reference configuration**, not exact numerical
ground truth; F32/F32 is a useful additional reference but is not proof
of a mathematically exact attention pipeline either.

## Kernel dispatch evidence — remains unverified

The benchmark **does not claim which CUDA attention kernel ran**.
The pinned binary contains vector Flash BF16/BF16 specializations, but
that does not prove use at prefill/decode.

To establish dispatch, a separate Nsight Systems/CUPTI capture of an
explicitly authorized Phase27 GPU worker is needed. Profile the two
phases separately, correlate device kernels by phase, and classify
actual VEC, MMA_F16 and TILE invocations. The current benchmark's
`native_kernel_dispatch_verified` remains `false`; prefill/decode
wall time by itself cannot prove a kernel choice or conversion.

No benchmark result automatically qualifies a runtime for production,
and the canonical Git profile remains unchanged until separately
reviewed numerical and kernel evidence exists.
