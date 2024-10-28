#include <core/Embedding.h>
#include <cpu/Embedding.h>

void embedding_forward_dispatch(
    const tensor_t *W,
    const tensor_t *input,
    tensor_t *output
) {
    device_t device = W->device;
    if (device == CPU) {
        embedding_forward_cpu_kernel(W, input, output);
    }
}

void embedding_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dW
) {
    device_t device = global_grad->device;
    if (device == CPU) {
        embedding_backward_cpu_kernel(global_grad, cache, dW);
    }
}