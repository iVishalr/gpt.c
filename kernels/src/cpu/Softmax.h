#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void softmax_forward_cpu_kernel(
    const tensor_t *input, tensor_t *output
);

#ifdef __cplusplus
}
#endif