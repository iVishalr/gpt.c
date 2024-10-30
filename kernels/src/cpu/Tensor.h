#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void move_tensor_to_host_cpu(tensor_t *tensor);
void move_tensor_to_device_cpu(tensor_t *tensor);

void create_tensor_data_cpu(tensor_t *tensor);
void zeros_tensor_data_cpu(tensor_t *tensor);
void ones_tensor_data_cpu(tensor_t *tensor);
void fill_tensor_data_cpu(tensor_t *tensor, const float value);

void copy_tensor_data_cpu(tensor_t *dst, const tensor_t *src);

void saxpy_cpu(
    const int n, const float alpha, 
    const tensor_t *x, const int offsetx, const int incx, 
    tensor_t *y, const int offsety, const int incy
);

#ifdef __cplusplus
}
#endif