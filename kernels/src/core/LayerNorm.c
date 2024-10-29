#include <core/LayerNorm.h>
#include <cpu/LayerNorm.h>

void layer_norm_forward_dispatch(
    const tensor_t *W,
    const tensor_t *b,
    const tensor_t *input,
    const float eps,
    tensor_t **cache,
    tensor_t *output
) {
    device_t device = input->device;
    if (device == CPU) {
        layer_norm_forward_cpu_kernel(W, b, input, eps, cache, output);
    }
}

void layer_norm_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t **cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
) {
    device_t device = global_grad->device;
    if (device == CPU) {
        layer_norm_backward_cpu_kernel(global_grad, cache, W, dW, db, dout);
    }
}