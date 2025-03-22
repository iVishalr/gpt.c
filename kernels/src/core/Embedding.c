#include <core/Embedding.h>
#include <cpu/Embedding.h>
#include <cuda/Embedding.h>
#include "utils.h"

void embedding_forward_dispatch(
    const tensor_t *W,
    const tensor_t *input,
    tensor_t *output
) {
    CHECK_ERROR(
        input->device != output->device,
        "Expected both input and output tensors to be on the same device, but got input.device != output.device"
    );
    CHECK_ERROR(
        input->device != W->device,
        "Expected both input and weight tensors to be on the same device, but got input.device != W.device"
    );
    device_t device = W->device;
    if (device == CPU)
        embedding_forward_cpu_kernel(W, input, output);
    else if (device == CUDA)
        embedding_forward_cuda_kernel(W, input, output);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void embedding_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dW
) {
    CHECK_ERROR(
        global_grad->device != dW->device,
        "Expected both glboal_grad and dW tensors to be on the same device, but got global_grad.device != dW.device"
    );
    device_t device = global_grad->device;
    if (device == CPU)
        embedding_backward_cpu_kernel(global_grad, cache, dW);
    else if (device == CUDA)
        embedding_backward_cuda_kernel(global_grad, cache, dW);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}