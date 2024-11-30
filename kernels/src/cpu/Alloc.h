#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *aligned_alloc_cpu(const size_t nbytes, const size_t alignment);

void free_cpu(void *ptr);

#ifdef __cplusplus
}
#endif