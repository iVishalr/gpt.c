#include <core/AdamW.h>
#include <cpu/AdamW.h>
#include <cuda/AdamW.h>
#include "utils.h"

void step_adamW_dispatch(
    tensor_t **parameters,
    tensor_t **gradients,
    tensor_t **m,
    tensor_t **v,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
) {
    // Check if all tensors are present on the same device.
    device_t device = parameters[0]->device;
    for (int i = 0; i < n; i++) {
        CHECK_ERROR(
            parameters[i]->device != device, 
            "Expected parameter at position %d of size %d to be on device %d, but got device = %d.", 
            i, parameters[i]->length, device, parameters[i]->device
        );
        CHECK_ERROR(
            gradients[i]->device != device, 
            "Expected gradient at position %d of size %d to be on device %d, but got device = %d.", 
            i, gradients[i]->length, device, gradients[i]->device
        );
        CHECK_ERROR(
            m[i]->device != device, 
            "Expected momentum1 (m) at position %d of size %d to be on device %d, but got device = %d.", 
            i, m[i]->length, device, m[i]->device
        );
        CHECK_ERROR(
            v[i]->device != device, 
            "Expected momentum2 (v) at position %d of size %d to be on device %d, but got device = %d.", 
            i, v[i]->length, device, v[i]->device
        );
    }
    if (device == CPU)
        step_adamW_cpu_kernel(parameters, gradients, m, v, n, lr, beta1, beta2, weight_decay, eps, step);
    else if (device == CUDA)
        step_adamW_cuda_kernel(parameters, gradients, m, v, n, lr, beta1, beta2, weight_decay, eps, step);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void zero_grad_adamW_dispatch(tensor_t **gradients, const int n) {
    device_t device = gradients[0]->device;
    for (int i = 0; i < n; i++) {
        CHECK_ERROR(
            gradients[i]->device != device, 
            "Expected gradient at position %d of size %d to be on device %d, but got device = %d.", 
            i, gradients[i]->length, device, gradients[i]->device
        );
    }
    if (device == CPU)
        zero_grad_adamW_cpu_kernel(gradients, n);
    else if (device == CUDA)
        zero_grad_adamW_cuda_kernel(gradients, n);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}