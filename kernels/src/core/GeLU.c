#include <core/GeLU.h>
#include <cpu/GeLU.h>

void gelu_forward_dispatch(
    const tensor_t *input,
    tensor_t *output
) {
    device_t device = input->device;
    gelu_forward_cpu_kernel(input, output);
}

void gelu_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dout
) {
    gelu_backward_cpu_kernel(global_grad, cache, dout);
}