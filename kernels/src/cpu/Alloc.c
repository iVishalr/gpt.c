#include <stdlib.h>
#include <cpu/Alloc.h>
#include "utils.h"

static inline size_t compute_aligned_tensor_size(const size_t nbytes, const size_t alignment) {
    return (nbytes + alignment - 1) & (~(alignment - 1));
}

void *aligned_alloc_cpu(const size_t nbytes, const size_t alignment) {
    const size_t aligned_nbytes = compute_aligned_tensor_size(nbytes, alignment);
    void *ptr = aligned_alloc(alignment, aligned_nbytes);
    return ptr;
}

void free_cpu(void *ptr) {
    if (ptr == NULL) 
        return;
    free(ptr);
}
