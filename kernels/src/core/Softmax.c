#include <core/Softmax.h>
#include <cpu/Softmax.h>

void softmax_forward_dispatch(
    const tensor_t *input,
    tensor_t *output)
{
    device_t device = input->device;
    softmax_forward_cpu_kernel(input, output);
}
