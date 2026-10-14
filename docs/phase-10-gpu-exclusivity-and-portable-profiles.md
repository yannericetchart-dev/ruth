# NVIDIA GPU exclusivity, VRAM accounting, and portable capacity profiles

## Why

Ruth must not conclude that a quantization is globally incompatible because one
RTX 4050 with driver 595.99.02 failed a CUDA allocation. An earlier
`feature/minimal-chat` load reportedly worked. This remains an open
regression to be tested after GPU isolation. No driver rollback, model switch,
context reduction, or CPU fallback is automatic.

## Probe preflight

`./ruth models probe-production --allow-gpu-inference`, GGUF smoke and
profiling call the read-only `gpu_preflight.inspect_gpu_preflight()` before
model worker startup. Ruth executes `nvidia-smi -q -x -i <GPU-UUID>`,
matches the selected UUID, reads framebuffer total/reserved/used/free,
and examines **both graphics and compute processes** from the process list.
The check is repeated immediately before the worker starts. If any process is
reported (including Xorg), status becomes `gpu_busy`; if telemetry is
incomplete, `gpu_preflight_unverified`. Neither case starts the worker or
publishes a successful profile. Ruth returns PID, process type and name,
GPU-reported process memory when available and host recommendations in
`gpu_preflight`. Ruth never kills host processes.

Process queries are a preflight snapshot, not a global OS-wide lock: new
third-party GPU users can still start between checks. Operators requiring
strict exclusivity should isolate the host environment.

## Host-side troubleshooting

```bash
nvidia-smi
nvidia-smi -q -d MEMORY
sudo fuser -v /dev/nvidia*
```

For Xorg on the selected NVIDIA device, first save all graphical work and
connect via TTY or SSH. To temporarily stop the graphical session:

```bash
sudo systemctl stop display-manager
nvidia-smi
nvidia-smi -q -d MEMORY
./ruth models probe-production --allow-gpu-inference
sudo systemctl start display-manager
```

This stops the graphical login/session and **must not** be run from a terminal
in that session. Use TTY or SSH and know how to restore your display manager.
Do not change the display manager or PRIME wiring automatically.

## Memory interpretation

Free memory already excludes reported reserved and used memory; **do not
subtract reserved a second time**. The 2026-10-14 example was total=6141 MiB,
reserved=369 MiB, used=15 MiB, free=5759 MiB, with Xorg using 4 MiB.
The failed allocation requested 5700.31 MiB, but separate CUDA runtime
allocations can still fail: free memory is not a promise that the requested
contiguous allocation will succeed. A controlled clean-host A/B comparison with
`feature/minimal-chat` is required before assigning the regression to driver
or runtime. CUDA-context-specific free memory is a later calibration measure,
not inferred from NVML.

## Catalog and profiles

`catalog/model_catalog.json` lists portable pinned GGUF properties and
neutral selection hints. Hardware outcomes live in runtime diagnostics and
local reports, not in that catalog. `catalog/gpu_capacity_profiles.json`
defines generic nominal 4, 6, 8, 10, 12, 16, 24, 32 and 48 GiB classes.
Unknown GPU memory sizes remain valid: always use detected bytes and current
free VRAM. Classes are not hard-coded compatibility claims. Only real full-GPU
load and generation certify a production candidate. Never automatically
downgrade quantization or fall back to CPU.

## Validation

Offline tests include:
`tests/test_phase10_gpu_preflight.py` and the production guard regression test.
Real GPU validation must run on the host only after removing all concurrent GPU
processes and must compare the same GGUF, wheel, native build and load options
against `feature/minimal-chat`.
