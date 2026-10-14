# Phase 12 — Portable model memory profiles and host capability matrix

## Objective

Ruth exposes three different, non-interchangeable concepts:

1. **Available**: a SHA-pinned GGUF exists in Ruth's neutral catalog.
   This does not mean it was downloaded or loaded on this host.
2. **Capacity projection**: measured CUDA model-weights buffer size can
   exclude physically insufficient VRAM classes, but passing that necessary
   condition is not proof that model+KV+compute will fit.
3. **Host verification**: a host-only probe ties its results to the exact
   GGUF SHA256, native binding version, GPU UUID, driver version and load
   parameters. A load-only probe is still NOT production inference proof.

Never record RTX/driver-specific failures as unconditional verdicts in
`catalog/model_catalog.json`. The list of nominal capacity classes is
hardware neutral in `catalog/gpu_capacity_profiles.json`: 4, 6, 8,
10, 12, 16, 24, 32 and 48 GiB. Unknown class GPUs can still be evaluated
from actual hardware measurements; nominal classes are not a whitelist.

## Read-only CLI

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling
.venv/bin/python -m unittest tests.test_phase12_model_memory_profiles -v

# No nvidia-smi, CUDA or GPU allocations. All pinned catalog profiles.
./ruth models profiles

# Read-only host GPU telemetry, GPU process exclusivity, DRM association
./ruth models profiles --host

# Locally save evidence from the real no-inference weights experiment.
# A failing native load exits nonzero; its JSON is still written.
./ruth models probe-load --load-mode weights --cuda-preinit \
  --allow-gpu-load > /tmp/ruth-weights-evidence.json || true

# Attach only this exact model+runtime+host evidence to the capacity matrix.
./ruth models profiles --host --quantization Q4_K_L \
  --evidence-file /tmp/ruth-weights-evidence.json

# Shows a separately-authorized experimental choice; DOES NOT execute a probe.
./ruth models profiles --host --quantization Q4_K_L \
  --evidence-file /tmp/ruth-weights-evidence.json --force
```

`--force` **cannot** bypass an active GPU process, a non-compute-only DRM
configuration, a missing GPU, pinned model integrity requirements,
runtime checks, or an explicit `--allow-gpu-load` gate for a later probe.
It changes **only the planning recommendation** and has no side effects.

A validated evidence file is read from disk but never copied into the
generic model catalog. Current version accepts a single report; accumulating
multi-context measurements into durable per-model local profiles is the
next step.

## Projection rule

For each registered GGUF and nominal GPU memory class:

- `weights_buffer_mib > nominal_total_mib`: insufficient *physical* VRAM
  for the same weights allocation — reject this particular full-GPU allocation.
- `weights_buffer_mib <= nominal_total_mib`: *potentially* possible, not
  verified; driver reserve, CUDA init, KV/attention and compute buffers
  may still make loading impossible.
- No validated native buffer measurement: unknown, do not replace it with
  the serialized GGUF filesize or manufacture a VRAM reserve.

On the host, use the GPU's `nvidia-smi` preflight for total/free/
reserved/used and GPU process checks. For a matching exact CUDA preinit
report, compare native weights-buffer demand against the **observed CUDA
free memory after context activation**. On this host the comparison is
5682.81 MiB available versus 5700.31 MiB requested. This is a measured
negative verdict for weights on this exact CUDA configuration; it is **not**
a generic claim that all 6 GiB GPUs must fail.

The resource profile that will ultimately enable positive host compatibility
needs **all** of the following: weights load confirmation; context/KV/compute
allocations with actual residency on CUDA; per-context `n_ctx`, `n_batch`,
`n_ubatch`, K/V precision and Flash Attention configuration; driver/runtime
provenance and live GPU exclusivity; then a separate inference verification
gate for production. No arbitrary fixed percentage VRAM margin.

## DRM check

`nvidia_drm` loaded in `lsmod` is **not sufficient** evidence that the
selected GPU is actively used for a DRM/KMS device. Ruth looks for
`/sys/class/drm/cardN/device` that resolves to the selected GPU PCI
device, independently reporting module-loaded and PCI device association.
It does not unload modules, terminate Xorg/Wayland or change drivers.
If sysfs cannot be verified, it reports uncertainty rather than passing
a compute-only check. Default profile listing still works when DRM is
active; compute-only experiment advice requires DRM-not-loaded.

The user has confirmed that even a clean boot blacklisting NVIDIA DRM and
modeset leaves the same 369 MiB driver reserve. A loaded graphics module
is therefore **not** equivalent to the cause of reserved VRAM.

## KV cache padding and Flash Attention research

The KV allocation is *not yet observed for Q4_K_L on the 6 GiB host*:
the model weights fail first. No KV-derived memory figure can be inferred
from that failure. Future probes must record the native `kv_size`,
K/V buffer MiB, KV dtype, actually selected flash-attention mode and CUDA
compute buffer at each context size (for example 128, 255, 256, 257,
511, 512, 513).

An older `llama.cpp` implementation uses **32 KV pad cells with FA off
and 256 with FA on** for kernel bounds/graph reuse; other versions
may additionally pad graph-effective KV to 256 regardless of physical KV
allocation. These are two different notions of padding. Ruth must measure
what the **pinned native backend** actually allocates rather than
hardcode 256 universally across versions or models. With quantized KV,
different `type_k`/`type_v` produce different step sizes. Flash Attention
can change intermediate compute/scratch buffers as well as cache layout.
Do not synthesize memory numbers before native measurement.

Reference: https://github.com/ggml-org/llama.cpp/discussions/13816

## Status semantics

- `insufficient_nominal_vram_for_measured_weights` — negative physical
  bound supported by a pinned native weights-buffer measurement.
- `weights_physically_possible_unverified` — no runtime guarantee.
- `native_memory_profile_pending` — missing native measurements.
- `host_measured_weights_cannot_fit` — measured CUDA free VRAM on matching
  GPU UUID/driver is less than this model's measured weights buffer.
- `host_context_allocation_observed_inference_unverified` — distinct from
  production qualified; a context allocation by itself is not generation.
- `host_evidence_identity_mismatch` — never transplant a host result as
  valid evidence for a different GPU/driver.

The advisor deliberately sets `production_qualified=false` everywhere.
Actual production GPU qualification is a separate, stricter Ruth pipeline.


### Interfaccia `--force`: disponibilità distinta da autorizzazione

`host_force_experiment_options` e `experiment_force_available` elencano
le prove sperimentali consentite dai controlli preliminari anche quando
`--force` **non** è stato richiesto. Invece
`experiment_force_requested` ed
`experiment_allowed_in_separate_probe` rimangono false finché l'utente
non passa esplicitamente `--force`. Nessuna delle due modalità esegue
CUDA: per questo occorre sempre un secondo comando autorizzato.

Un profilo con `host_measured_weights_cannot_fit` può comunque mostrare
l'opzione sperimentale (per raccogliere nuove prove dopo modifiche reali
all'ambiente); l'opzione non nega né altera la diagnosi negativa corrente.
