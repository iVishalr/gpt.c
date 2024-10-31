#include <core/Attention.h>
#include <cpu/Attention.h>

void attention_forward_dispatch(
    const tensor_t *input,
    const tensor_t *mask,
    const int n_heads,
    tensor_t **cache,
    tensor_t *output
) {
    device_t device = input->device;
    if (device == CPU) {
        attention_forward_cpu_kernel(input, mask, n_heads, cache, output);
    }
}

void attention_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t **cache,
    const int n_heads,
    tensor_t *dout
) {
    device_t device = global_grad->device;
    if (device == CPU) {
        attention_backward_cpu_kernel(global_grad, cache, n_heads, dout);
    }
}