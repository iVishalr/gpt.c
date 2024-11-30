#include <core/Softmax.h>
#include <cpu/Softmax.h>
#include <cuda/Softmax.h>
#include "utils.h"

void softmax_forward_dispatch(
    const tensor_t *input,
    tensor_t *output
) {
    CHECK_ERROR(
        input->device != output->device,
        "Expected both input and output tensors to be on the same device, but got input.device != output.device"
    );
    device_t device = input->device;
    if (device == CPU)
        softmax_forward_cpu_kernel(input, output);
    else if (device == CUDA)
        softmax_forward_cuda_kernel(input, output);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void softmax_backward_dispatch(
    const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout
) {
    CHECK_ERROR(
        global_grad->device != dout->device,
        "Expected both global_grad and dout tensors to be on the same device, but got global_grad.device != dout.device"
    );
    device_t device = global_grad->device;
    if (device == CPU)
        softmax_backward_cpu_kernel(global_grad, cache, dout);
    else if (device == CUDA)
        softmax_backward_cuda_kernel(global_grad, cache, dout);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}