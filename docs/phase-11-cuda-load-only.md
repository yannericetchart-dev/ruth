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
