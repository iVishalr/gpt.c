#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void softmax_forward_dispatch(
    const tensor_t *input,
    tensor_t *output
);

#ifdef __cplusplus
}
#endif