#include <core/CrossEntropyLoss.h>
#include <cpu/CrossEntropyLoss.h>
#include "utils.h"

void cross_entropy_forward_dispatch(
    const tensor_t *logits,
    const tensor_t *targets,
    tensor_t **cache,
    tensor_t *output
) {
    CHECK_ERROR(
        logits->device != targets->device,
        "Expected both probs and targets tensors to be on the same device, but got probs.device != targets.device"
    );
    CHECK_ERROR(
        logits->device != output->device, 
        "Expected both probs and output tensors to be on the same device, but got probs.device != output.device"
    );
    device_t device = output->device;
    if (device == CPU)
        cross_entropy_forward_cpu_kernel(logits, targets, cache, output);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void cross_entropy_backward_dispatch(
    const tensor_t *global_grad,
    const tensor_t **cache,
    tensor_t *dout
) {
    CHECK_ERROR(
        global_grad->device != dout->device, 
        "Expected both logits and targets tensors to be on the same device, but got logits.device != targets.device"
    );
    device_t device = global_grad->device;
    if (device == CPU)
        cross_entropy_backward_cpu_kernel(global_grad, cache, dout);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}