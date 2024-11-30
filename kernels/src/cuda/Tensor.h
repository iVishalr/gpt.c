#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void move_tensor_to_host_cuda(tensor_t *tensor);
void move_tensor_to_device_cuda(tensor_t *tensor);

// void create_tensor_data_cpu(tensor_t *tensor);
// void zeros_tensor_data_cpu(tensor_t *tensor);
// void ones_tensor_data_cpu(tensor_t *tensor);
// void fill_tensor_data_cpu(tensor_t *tensor, const float value);

// void copy_tensor_data_cpu(tensor_t *dst, const tensor_t *src);

// void saxpy_cpu(
//     const int n, const float alpha, 
//     const tensor_t *x, const int offsetx, const int incx, 
//     tensor_t *y, const int offsety, const int incy
// );

// void sgemm_cpu(
//     const int TransA, const int TransB, const int M, const int N, const int K,
//     const float alpha, const tensor_t *A, const int offsetA, const int lda,
//     const tensor_t *B, const int offsetB, const int ldb, 
//     const float beta, tensor_t *C, const int offsetC, const int ldc
// );

#ifdef __cplusplus
}
#endif