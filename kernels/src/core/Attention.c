#include <core/Attention.h>
#include <cpu/Attention.h>
#include <cuda/Attention.h>
#include "utils.h"

void attention_forward_dispatch(
    const tensor_t *input,
    const tensor_t *mask,
    const int n_heads,
    tensor_t **cache,
    tensor_t *output
) {
    CHECK_ERROR(
        input->device != output->device,
        "Expected both input and output tensors to be on the same device, but got input.device != output.device"
    );
    CHECK_ERROR(
        input->device != mask->device,
        "Expected both input and mask tensors to be on the same device, but got input.device != mask.device"
    );
    CHECK_ERROR(
        input->device != cache[0]->device,
        "Expected both input and output tensors to be on the same device, but got input.device != cache[0].device"
    );
    CHECK_ERROR(
        input->device != cache[1]->device,
        "Expected both input and output tensors to be on the same device, but got input.device != cache[1].device"
    );    
    CHECK_ERROR(
        input->device != cache[2]->device,
        "Expected both input and output tensors to be on the same device, but got input.device != cache[2].device"
    );
    CHECK_ERROR(
        input->device != cache[3]->device,
        "Expected both input and output tensors to be on the same device, but got input.device != cache[3].device"
    );
    device_t device = input->device;
    if (device == CPU)
        attention_forward_cpu_kernel(input, mask, n_heads, cache, output);
    else if (device == CUDA)
        attention_forward_cuda_kernel(input, mask, n_heads, cache, output);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void attention_backward_dispatch(
    const tensor_t *global_grad,
    tensor_t **cache,
    const int n_heads,
    tensor_t *dout
) {
    CHECK_ERROR(
        global_grad->device != dout->device,
        "Expected both input and output tensors to be on the same device, but got global_grad.device != dout.device"
    );
    CHECK_ERROR(
        global_grad->device != cache[0]->device,
        "Expected both input and output tensors to be on the same device, but got global_grad.device != cache[0].device"
    );
    CHECK_ERROR(
        global_grad->device != cache[1]->device,
        "Expected both input and output tensors to be on the same device, but got global_grad.device != cache[1].device"
    );    
    CHECK_ERROR(
        global_grad->device != cache[2]->device,
        "Expected both input and output tensors to be on the same device, but got global_grad.device != cache[2].device"
    );
    CHECK_ERROR(
        global_grad->device != cache[3]->device,
        "Expected both input and output tensors to be on the same device, but got global_grad.device != cache[3].device"
    );
    device_t device = global_grad->device;
    if (device == CPU)
        attention_backward_cpu_kernel(global_grad, cache, n_heads, dout);
    else if (device == CUDA)
        attention_backward_cuda_kernel(global_grad, cache, n_heads, dout);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}