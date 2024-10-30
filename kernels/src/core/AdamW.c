#include <core/AdamW.h>
#include <cpu/AdamW.h>

void step_adamW_dispatch(
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
)
{
    step_adamW_pytorch_cpu_kernel(parameters, gradients, m, v, n, lr, beta1, beta2, weight_decay, eps, step);
}

void zero_grad_adamW_dispatch(tensor_t **gradients, const int n) {
    zero_grad_adamW_cpu_kernel(gradients, n);
}