#include <core/Linear.h>
#include <cpu/Linear.h>
#include <cuda/Linear.h>
#include "utils.h"

void linear_forward_dispatch(
    const tensor_t *W,
    const tensor_t *b,
    const tensor_t *input,
    tensor_t *output)
{
    CHECK_ERROR(
        input->device != output->device,
        "Expected both input and output tensors to be on the same device, but got input.device != output.device"
    );
    CHECK_ERROR(
        input->device != W->device,
        "Expected both input and weight tensors to be on the same device, but got input.device != weight.device"
    );
    if (b != NULL)
        CHECK_ERROR(
            input->device != b->device,
            "Expected both input and bias tensors to be on the same device, but got input.device != bias.device"
        );
    device_t device = input->device;
    if (device == CPU)
        linear_forward_cpu_kernel(W, b, input, output);
    else if (device == CUDA)
        linear_forward_cuda_kernel(W, b, input, output);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void linear_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t *cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
) {
    CHECK_ERROR(
        global_grad->device != dout->device,
        "Expected both global_grad and dout tensors to be on the same device, but got global_grad.device != dout.device"
    );
    CHECK_ERROR(
        global_grad->device != W->device,
        "Expected both global_grad and weight tensors to be on the same device, but got input.device != weight.device"
    );
    CHECK_ERROR(
        global_grad->device != dW->device,
        "Expected both input and dW tensors to be on the same device, but got input.device != dW.device"
    );
    if (db != NULL)
        CHECK_ERROR(
            global_grad->device != db->device,
            "Expected both input and db tensors to be on the same device, but got input.device != db.device"
        );
    device_t device = global_grad->device;
    if (device == CPU)
        linear_backward_cpu_kernel(global_grad, cache, W, dW, db, dout);
    else if (device == CUDA)
        linear_backward_cuda_kernel(global_grad, cache, W, dW, db, dout);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}