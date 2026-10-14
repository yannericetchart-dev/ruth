# Phase 11: Native CUDA load and buffer allocation, **without inference**

Aim: separate the model-weight CUDA buffer from KV/compute allocation on a
pinned Gemma 2 9B GGUF, using the same installed CUDA `llama-cpp-python`
binding. This is not a production qualification, benchmark or prediction of
available context. It NEVER tokenizes, invokes `llama_decode`, evaluates a
prompt, generates tokens, or publishes a runtime profile.

## Commands (project venv)

```bash
cd ~/Dev/ruth-model-profiling
git pull --ff-only origin model-profiling
.venv/bin/python -m unittest discover -s tests -v

# Stage A: native model weights only — NO context / KV / inference
./ruth models probe-load --load-mode weights --allow-gpu-load

# Stage B: after weights pass, allocate KV/compute with no inference
./ruth models probe-load --load-mode context --n-ctx 128 --allow-gpu-load
```

To inspect any suspected padding/step around 256 tokens, repeat **context
mode** at 128, 255, 256 and 257, only after the weight stage succeeds. Every
run is an isolated worker and starts from fresh GPU memory. Compare
`native_kv_sizes`, `native_buffers`,
`cuda_free_mib_after_model_load` and
`cuda_free_mib_after_context_init`. For an OOM in the weights phase, do not
repeat context scans: none can reach KV allocation.

```bash
./ruth models probe-load --load-mode context --n-ctx 255 --allow-gpu-load
./ruth models probe-load --load-mode context --n-ctx 256 --allow-gpu-load
./ruth models probe-load --load-mode context --n-ctx 257 --allow-gpu-load
```

`gpu_preflight` requires no graphics or compute process on the selected
NVIDIA GPU. Read-only sampling and `cuMemGetInfo_v2` gather available
memory; the latter reads a **current already initialized CUDA context** and
does not create an additional context. When unavailable, the field is null.
Native stderr is parsed for reported model/KV/compute/output buffer sizes and
OOM causes; the brief native excerpts are redacted. The worker uses
`llama_model_load_from_file`, then optionally `llama_init_from_model`.
It never calls inference APIs. No automatic quantization changes, CPU fallback,
or network/download actions occur.

## Why 256 might appear

Certain `llama.cpp` versions round KV allocations for Flash Attention to
256 tokens (and to 32 otherwise), and computation may use separate 256-token
alignment. This originates from `llama.cpp` kernel/alignment behavior,
not necessarily a fixed Gemma-specific increment or a universally fixed
allocation policy. Only the runtime's native `kv_size` and GPU buffer logs
tell us what the installed build actually allocated. The failed 5700.31 MiB
`ggml_backend_cuda_buffer_type_alloc_buffer` is a **model weights buffer**
allocation: a lower `n_ctx` cannot directly shrink that weights buffer.

## Driver headless hypothesis

The driver 595.99.02 already reports `reserved=369 MiB` under both X11 and
Wayland with no NVIDIA processes under Wayland. Do not blacklist
`nvidia_drm`/`nvidia_modeset` or set undocumented interrupt registry
options as part of the load probe. Stopping DRM/KMS may remove graphics
capabilities and is not proven to reduce the driver reservation. If explored,
keep it an isolated, reversible host-side experiment from TTY/SSH and confirm
that Intel Wayland and CUDA still work. Avoid a simultaneous driver change,
runtime upgrade or quantization change.

`feature/minimal-chat` presently contains tensor-level allocation overrides;
the user also reports that its earlier default, unmodified loading configuration
succeeded. Both pieces of history must be kept separate: do not attribute
this failure to overrides, Python versus C API, or driver update alone without
a controlled comparison.

## Additional controlled A/B: CUDA primary-context preinitialization

The clean headless baseline (GPU processes 0; reserved=369 MiB, free=5773 MiB)
still fails with CUDA `cudaMalloc` asking for a 5700.31 MiB model buffer.
`cuMemGetInfo` only after failure reports 5682.81 MiB free; that value
**is not** proof of the exact free memory at the instant of the failed call.

An explicitly separate `--cuda-preinit` diagnostic retains and binds
device-0's **primary CUDA context** via CUDA Driver API before model load,
records `cuda_free_mib_after_primary_ctx_init` in the *same process*, then
attempts the **same native weights-only GGUF load**. All additional context
references are released during worker cleanup; no token generation occurs.

```bash
.venv/bin/python -m unittest tests.test_phase11_load_only_probe -v
./ruth models probe-load --load-mode weights --allow-gpu-load --cuda-preinit
```

Compare the optional `cuda_free_mib_after_primary_ctx_init` to the required
5700.31 MiB. This is a **controlled intervention**, not an exact passive
measurement in the untouched loading path: explicitly preinitializing CUDA
can itself alter context setup order or allocator behavior. Interpret alongside
the earlier standard `--load-mode weights` baseline; a causal claim requires
additional comparison. No driver restart, module blacklist, quantization
switch, or CPU fallback is part of this diagnostic.

## Host result: explicit CUDA primary context (2026-10-14)

On the RTX 4050 6141 MiB with NVIDIA driver 595.99.02, in a clean
`module_blacklist=nvidia_drm,nvidia_modeset` boot, the user ran the
`weights --cuda-preinit` path. All 11 isolated regression tests passed.
The native CUDA driver API returned **5682.81 MiB free after activating
the primary context, before the GGUF model-load call**. NVIDIA-SMI reported
5773 MiB free before the worker. Thus the measured difference across the
two APIs/times is 90.19 MiB. The GGUF model CUDA buffer requested
**5700.31 MiB**, exceeding the pre-load measurement by **17.50 MiB**.
The model-load API returned null, the native trace reported `cudaMalloc
out of memory`, `weights_loaded=false`, `context_initialized=false`,
`generated_tokens=0`; no context/KV or inference step ran.

This specifically establishes insufficient CUDA free memory for the reported
weights buffer in the explicit primary-context experiment. It does not by
itself prove the *cause* of all 369 MiB driver-reserved memory, nor the
size/sequence of possible later KV/compute allocations. Avoid claiming
the difference in separate NVML and CUDA samples is an exact breakdown
of CUDA context allocations.

## Reconciliation with historical `feature/minimal-chat`

The historical branch `model/manager.py` sets its **default file to
`gemma-2-9b-it-Q4_K_M.gguf`**. The early
`calibration/probe.py` starts with **`--gpu-layers 20`** when that argument
is omitted; it passes that layer count to `Llama(n_gpu_layers=...)`.
Conversely, the `model-profiling` default is Q4_K_L with full layer
offload requested. For the pinned files currently in the neutral catalog:

- Q4_K_L: 5983265728 serialized bytes.
- Q4_K_M: 5761057728 serialized bytes.
- File-size difference: 222208000 bytes, approximately 211.91 MiB.

Thus a historically successful *default* load is **not necessarily an
equivalent A/B** of Q4_K_L, full GPU weights, on the current wheel and
driver. Later old-branch experiments use full-GPU tensor overrides and
explicit GGUF baseline manifests, which must be evaluated separately.

To reproduce the old default as an **explicit diagnostic only**, with
full GPU offload still requested and no inference, use:

```bash
./ruth models probe-load --quantization Q4_K_M --load-mode weights --allow-gpu-load
```

This requires the separate, pinned Q4_K_M GGUF to be locally available
and verified; otherwise the command fails closed. Do not download
anything or change the production default as part of this test. A
successful Q4_K_M weights load would not prove the Q4_K_L load can
succeed, and does not certify that KV/compute buffers also fit.
