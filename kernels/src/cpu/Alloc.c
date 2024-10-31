#include <stdlib.h>
#include <cpu/Alloc.h>
#include "utils.h"

size_t compute_aligned_tensor_size(const size_t nbytes, const size_t alignment) {
    return (nbytes + alignment - 1) & (~(alignment - 1));
}

void *aligned_alloc_cpu(const size_t nbytes, const size_t alignment) {
    const size_t aligned_nbytes = compute_aligned_tensor_size(nbytes, alignment);
    void *ptr = aligned_alloc(alignment, aligned_nbytes);
    return ptr;
}

void free_cpu(float *ptr) {
    if (ptr == NULL) 
        return;
    free(ptr);
}

tensor_t *as_tensor(float *data, const int *shape, const int n) {
    tensor_t *tensor = (tensor_t*)mallocCheck(sizeof(tensor_t));
    tensor->t = data;
    int length = 1;
    for (int i = 0; i < n; i++) {
        length *= shape[i];
        tensor->shape[i] = shape[i];
    }
    tensor->length = length;
    tensor->ndims = n;
    tensor->device = CPU;
    tensor->to = NULL;
    return tensor;
}