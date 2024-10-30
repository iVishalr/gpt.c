#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

void step_adamW_naive_cpu_kernel(
    tensor_t **parameters,
    tensor_t **gradients,
    tensor_t **m,
    tensor_t **v,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
);

void step_adamW_pytorch_cpu_kernel(
    tensor_t **parameters,
    tensor_t **gradients,
    tensor_t **m,
    tensor_t **v,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
);

void zero_grad_adamW_cpu_kernel(tensor_t **gradients, const int n);

#ifdef __cplusplus
}
#endif