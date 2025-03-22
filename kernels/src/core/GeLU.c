#include <core/GeLU.h>
#include <cpu/GeLU.h>
#include <cuda/GeLU.h>

void gelu_forward_dispatch(
    const tensor_t *input,
    tensor_t *output
) {
    device_t device = input->device;
    if (device == CPU)
        gelu_forward_cpu_kernel(input, output);
    else if (device == CUDA)
        gelu_forward_cuda_kernel(input, output);
}

void gelu_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dout
) {
    device_t device = global_grad->device;
    if (device == CPU)
        gelu_backward_cpu_kernel(global_grad, cache, dout);
    else if (device == CUDA)
        gelu_backward_cuda_kernel(global_grad, cache, dout);
}