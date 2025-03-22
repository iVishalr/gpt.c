#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void linear_forward_dispatch(
    const tensor_t *W,
    const tensor_t *b,
    const tensor_t *input,
    tensor_t *output
);

void linear_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
);

// void linear_backward_dispatch();

#ifdef __cplusplus
}
#endif