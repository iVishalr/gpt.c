#include <core/Alloc.h>
#include <cpu/Alloc.h>
#include <cuda/Alloc.h>
#include "utils.h"

void *alloc_dispatch(const size_t nbytes, const size_t alignment, const device_t device) {
    void *allocation = NULL;
    if (device == CPU) {
        allocation = aligned_alloc_cpu(nbytes, alignment);
    } else if (device == CUDA) {
        allocation = alloc_cuda(nbytes);
    } else {
        CHECK_ERROR(1, "Failed to allocate memory. Unknow device.");
    }
    return allocation;
}

void free_dispatch(float *ptr, const device_t device) {
    if (device == CPU) {
        free_cpu(ptr);
    } else if (device == CUDA) {
        free_cuda(ptr);
    } else {
        CHECK_ERROR(1, "Failed to allocate memory. Unknow device.");
    }
}