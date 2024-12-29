#pragma once

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

void move_tensor_to_host_cuda(tensor_t *tensor);
void move_tensor_to_device_cuda(tensor_t *tensor);

void create_tensor_data_cuda(tensor_t *tensor);
void zeros_tensor_data_cuda(tensor_t *tensor);
void ones_tensor_data_cuda(tensor_t *tensor);
void fill_tensor_data_cuda(tensor_t *tensor, const float value);
void arange_tensor_data_cuda(tensor_t *tensor, const int start, const int end, const int steps);
void copy_tensor_data_cuda(tensor_t *dst, const tensor_t *src);
void add_tensor_data_dispatch(tensor_t *dest, tensor_t *src);
void saxpy_cuda(
    const int n, const float alpha, 
    const tensor_t *x, const int offsetx, const int incx, 
    tensor_t *y, const int offsety, const int incy
);

void sgemm_cuda(
    const int TransA, const int TransB, const int M, const int N, const int K,
    const float alpha, const tensor_t *A, const int offsetA, const int lda,
    const tensor_t *B, const int offsetB, const int ldb, 
    const float beta, tensor_t *C, const int offsetC, const int ldc
);

void sgemm_strided_batched_cuda(
    const int TransA, const int TransB, const int M, const int N, const int K,
    const float alpha, const tensor_t *A, const int lda, const int strideA,
    const tensor_t *B, const int ldb, const int strideB,
    const float beta, tensor_t *C, const int ldc, const int strideC, const int batch_count
);

#ifdef __cplusplus
}
#endif