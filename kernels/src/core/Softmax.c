#include <core/Softmax.h>
#include <cpu/Softmax.h>

void softmax_forward_dispatch(
    const tensor_t *input,
    tensor_t *output
) {
    device_t device = input->device;
    softmax_forward_cpu_kernel(input, output);
}

void softmax_backward_dispatch(
    const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout
) {
    device_t device = global_grad->device;
    softmax_backward_cpu_kernel(global_grad, cache, dout);
}