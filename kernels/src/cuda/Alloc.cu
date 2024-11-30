#include <cuda/cuda_common.h>
#include <cuda/Alloc.h>

#ifdef __cplusplus
extern "C" {
#endif

void *alloc_cuda(const size_t size) {
    void *ptr;
    cudaCheck(cudaMalloc((void**)&ptr, size));
    return ptr;
}

void free_cuda(void *ptr) {
    if (ptr == NULL) return;
    cudaCheck(cudaFree(ptr));
}

#ifdef __cplusplus
}
#endif
