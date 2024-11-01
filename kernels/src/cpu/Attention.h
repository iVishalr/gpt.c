#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void attention_forward_cpu_kernel(
    const tensor_t *input,
    const tensor_t *mask,
    const int n_heads,
    const tensor_t **cache,
    tensor_t *output
);

void attention_backward_cpu_kernel(
    const tensor_t *global_grad, 
    const tensor_t **cache,
    const int n_heads,
    tensor_t *dout
);

#ifdef __cplusplus
}
#endif