#include <core/Linear.h>
#include <cpu/Linear.h>

void linear_forward_dispatch(
    const tensor_t *W,
    const tensor_t *b,
    const tensor_t *input,
    tensor_t *output)
{
    device_t device = W->device;
    // void *kernel_fn = NULL;
    // if (device == CPU) {
    //     kernel_fn = linear_forward_cpu_kernel;
    // } else if (device == CUDA) {
    //     kernel_fn = NULL;
    // }

    linear_forward_cpu_kernel(W, b, input, output);
}

void linear_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
) {
    linear_backward_cpu_kernel(global_grad, cache, W, dW, db, dout);
}