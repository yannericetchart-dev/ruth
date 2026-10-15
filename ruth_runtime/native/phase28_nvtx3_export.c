/* Optional Ruth Phase28 NVTX3 bridge.
 * NVTX3 is header-only. Explicitly export its C API for ctypes; no
 * runtime compilation, package installation, or CUDA initialization.
 * See https://github.com/NVIDIA/NVTX/blob/dev/docs/index.html
 * Compile manually outside the inference worker:
 * cc -std=c11 -O2 -fPIC -shared -I/usr/local/cuda/include \
 *   -o .ruth-build/phase28/libruth_phase28_nvtx3.so \
 *   ruth_runtime/native/phase28_nvtx3_export.c -ldl
 */
#define NVTX_EXPORT_API
#include <nvtx3/nvToolsExt.h>
