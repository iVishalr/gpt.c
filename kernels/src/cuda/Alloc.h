#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *alloc_cuda(const size_t nbytes);

void free_cuda(void *ptr);

#ifdef __cplusplus
}
#endif