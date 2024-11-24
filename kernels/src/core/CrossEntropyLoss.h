#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void cross_entropy_forward_dispatch(
    const tensor_t *logits,
    const tensor_t *targets,
    tensor_t **cache,
    tensor_t *output
);

void cross_entropy_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t **cache,
    tensor_t *dout
);

#ifdef __cplusplus
}
#endif