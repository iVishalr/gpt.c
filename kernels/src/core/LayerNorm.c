#include <core/LayerNorm.h>
#include <cpu/LayerNorm.h>
#include <cuda/LayerNorm.h>
#include "utils.h"

void layer_norm_forward_dispatch(
    const tensor_t *W,
    const tensor_t *b,
    const tensor_t *input,
    const float eps,
    tensor_t **cache,
    tensor_t *output
) {
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
        layer_norm_forward_cpu_kernel(W, b, input, eps, cache, output);
    else if (device == CUDA)
        layer_norm_forward_cuda_kernel(W, b, input, eps, cache, output);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void layer_norm_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t **cache,
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
        layer_norm_backward_cpu_kernel(global_grad, cache, W, dW, db, dout);
    else if (device == CUDA)
        layer_norm_backward_cuda_kernel(global_grad, cache, W, dW, db, dout);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}