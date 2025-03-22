#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/Alloc.h>

#ifdef __cplusplus
extern "C" {
#endif

void *alloc_cuda(const size_t size) {
    void *ptr;
    cudaStream_t stream = get_cuda_stream();
    cudaCheck(cudaMallocAsync((void**)&ptr, size, stream));
    return ptr;
}

void free_cuda(void *ptr) {
    if (ptr == NULL) return;
    cudaStream_t stream = get_cuda_stream();
    cudaCheck(cudaFreeAsync(ptr, stream));
}

#ifdef __cplusplus
}
#endif
