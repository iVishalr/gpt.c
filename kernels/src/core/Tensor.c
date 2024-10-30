#include <core/Tensor.h>
#include <cpu/Tensor.h>
#include "utils.h"

void create_tensor_data_dispatch(tensor_t *tensor) {
    device_t device = tensor->device;
    if (device == CPU)  
        create_tensor_data_cpu(tensor);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void zeros_tensor_data_dispatch(tensor_t *tensor) {
    device_t device = tensor->device;
    if (device == CPU)  
        zeros_tensor_data_cpu(tensor);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void ones_tensor_data_dispatch(tensor_t *tensor) {
    device_t device = tensor->device;
    if (device == CPU)  
        ones_tensor_data_cpu(tensor);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void fill_tensor_data_dispatch(tensor_t *tensor, const float value) {
    device_t device = tensor->device;
    if (device == CPU)  
        fill_tensor_data_cpu(tensor, value);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void copy_tensor_data_dispatch(tensor_t *dst, const tensor_t *src) {
    CHECK_ERROR(
        dst->device != src->device, 
        "Expected both source and destination tensors to be on the same device, but got dst.device != src.device"
    );
    device_t device = dst->device;
    if (device == CPU)
        copy_tensor_data_cpu(dst, src);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void saxpy_dispatch(
    const int n, const float alpha, 
    const tensor_t *x, const int offsetx, const int incx, 
    tensor_t *y, const int offsety, const int incy
) {
    CHECK_ERROR(
        x->device != y->device, 
        "Expected both x and y tensors to be on the same device, but got x.device != y.device"
    );
    device_t device = y->device;
    if (device == CPU)
        saxpy_cpu(n, alpha, x, offsetx, incx, y, offsety, incy);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}