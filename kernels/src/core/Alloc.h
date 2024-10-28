#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *alloc_dispatch(
    const size_t nbytes, const size_t alignment, const device_t device
);

void free_dispatch(
    void *ptr, const device_t device
);

#ifdef __cplusplus
}
#endif