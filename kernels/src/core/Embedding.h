#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void embedding_forward_dispatch(
    const tensor_t *W,
    const tensor_t *input,
    tensor_t *output
);

void embedding_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dW
);

#ifdef __cplusplus
}
#endif