#pragma once

#include "utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

void *AllocCheck(void *(*alloc_fn)(const size_t nbytes, const size_t alignment), const size_t nbytes, const size_t alignment);

#ifdef __cplusplus
}
#endif