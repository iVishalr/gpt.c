#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void attention_forward_dispatch(
    const tensor_t *input,
    const tensor_t *mask,
    const int n_heads,
    tensor_t **cache,
    tensor_t *output
);

void attention_backward_dispatch(
    const tensor_t *global_grad,
    tensor_t **cache,
    const int n_heads,
    tensor_t *dout
);

#ifdef __cplusplus
}
#endif