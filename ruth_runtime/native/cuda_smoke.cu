// Ruth phase 02: minimal, deterministic CUDA execution proof on sm_89.
// No models, no Python dependencies, no disk cache or performance claims.
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call, step)                                                   \
    do {                                                                         \
        cudaError_t status = (call);                                             \
        if (status != cudaSuccess) {                                             \
            std::fprintf(stderr, "RUTH_CUDA_ERROR step=%s code=%d message=%s\n", \
                         step, static_cast<int>(status),                        \
                         cudaGetErrorString(status));                           \
            return 20;                                                           \
        }                                                                        \
    } while (0)

__global__ void ruth_kernel(int *values, int count) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        values[i] = 2 * i + 3;
    }
}

int main() {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices), "get_device_count");
    if (devices < 1) {
        std::fprintf(stderr, "RUTH_CUDA_ERROR step=no_devices\n");
        return 21;
    }

    const int selected_device = 0; // Physical identity checked by Python via PCI.
    CUDA_CHECK(cudaSetDevice(selected_device), "set_device");
    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, selected_device), "device_properties");

    if (props.major != 8 || props.minor != 9) {
        std::fprintf(stderr,
                     "RUTH_CUDA_ERROR step=architecture expected=8.9 got=%d.%d\n",
                     props.major, props.minor);
        return 22;
    }

    char bus[32] = {};
    CUDA_CHECK(cudaDeviceGetPCIBusId(bus, sizeof(bus), selected_device),
               "pci_bus_id");

    constexpr int count = 256;
    constexpr size_t bytes = sizeof(int) * count;
    int* device_values = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_values), bytes),
               "cuda_malloc");

    ruth_kernel<<<2, 128>>>(device_values, count);
    cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        std::fprintf(stderr, "RUTH_CUDA_ERROR step=launch code=%d message=%s\n",
                     static_cast<int>(launch), cudaGetErrorString(launch));
        cudaFree(device_values);
        return 23;
    }
    cudaError_t sync = cudaDeviceSynchronize();
    if (sync != cudaSuccess) {
        std::fprintf(stderr, "RUTH_CUDA_ERROR step=synchronize code=%d message=%s\n",
                     static_cast<int>(sync), cudaGetErrorString(sync));
        cudaFree(device_values);
        return 24;
    }

    int host_values[count] = {};
    cudaError_t copy = cudaMemcpy(host_values, device_values, bytes,
                                  cudaMemcpyDeviceToHost);
    cudaError_t release = cudaFree(device_values);
    if (copy != cudaSuccess || release != cudaSuccess) {
        std::fprintf(stderr,
                     "RUTH_CUDA_ERROR step=copy_or_free copy=%s free=%s\n",
                     cudaGetErrorString(copy), cudaGetErrorString(release));
        return 25;
    }

    for (int i = 0; i < count; ++i) {
        if (host_values[i] != 2 * i + 3) {
            std::fprintf(stderr,
                         "RUTH_CUDA_ERROR step=validate index=%d expected=%d got=%d\n",
                         i, 2 * i + 3, host_values[i]);
            return 26;
        }
    }

    // Strict, machine-readable success token. No success until GPU computation
    // and host-side validation have both completed.
    std::printf("RUTH_CUDA_SMOKE_OK device=%d arch=%d.%d pci=%s elements=%d\n",
                selected_device, props.major, props.minor, bus, count);
    return 0;
}
