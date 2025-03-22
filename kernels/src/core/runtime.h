#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void runtime_init_dispatch(const device_t device);
void runtime_destroy_dispatch(const device_t device);
void synchronize_dispatch(const device_t device);

#ifdef __cplusplus
}
#endif