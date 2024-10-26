#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void gelu_forward_dispatch(
    const tensor_t *input,
    tensor_t *output
);

void gelu_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dout
);

#ifdef __cplusplus
}
#endif