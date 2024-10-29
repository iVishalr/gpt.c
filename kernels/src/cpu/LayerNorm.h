#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void layer_norm_forward_cpu_kernel(
    const tensor_t *W, 
    const tensor_t *b,
    const tensor_t *input,
    const float eps,
    tensor_t **cache,
    tensor_t *output
);

void layer_norm_backward_cpu_kernel(
    const tensor_t *global_grad,
    const tensor_t **cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
);

#ifdef __cplusplus
}
#endif