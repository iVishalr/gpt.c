#include <core/Tensor.h>
#include <cpu/Tensor.h>
#include <cuda/Tensor.h>
#include "utils.h"

void create_tensor_data_dispatch(tensor_t *tensor) {
    device_t device = tensor->device;
    if (device == CPU)  
        create_tensor_data_cpu(tensor);
    else if (device == CUDA)
        create_tensor_data_cuda(tensor);
    else
        CHECK_ERROR(1, "Given device is not supported.");
}

void zeros_tensor_data_dispatch(tensor_t *tensor) {
    device_t device = tensor->device;
    if (device == CPU)  
        zeros_tensor_data_cpu(tensor);
    else if (device == CUDA)
        zeros_tensor_data_cuda(tensor);
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

void move_tensor_data_dispatch(tensor_t *tensor, const device_t device) {
    const device_t src_device = tensor->device;
    int to_host = 0, to_device = 0;
    if (src_device != device && device != CPU) to_device = 1;
    else if (src_device != device && device == CPU) to_host = 1;
    if (to_host) {
        switch (src_device)
        {
        case CPU:
            move_tensor_to_host_cpu(tensor);
            break;
        case CUDA:
            move_tensor_to_host_cuda(tensor);
        default:
            break;
        }
    } else if (to_device) {
        switch (device)
        {
        case CPU:
            move_tensor_to_device_cpu(tensor);
            break;
        case CUDA:
            move_tensor_to_device_cuda(tensor);
        default:
            break;
        }
    }
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

void sgemm_dispatch(
    const int TransA, const int TransB, const int M, const int N, const int K,
    const float alpha, const tensor_t *A, const int offsetA, const int lda,
    const tensor_t *B, const int offsetB, const int ldb,
    const float beta, tensor_t *C, const int offsetC, const int ldc
) {
    CHECK_ERROR(
        A->device != B->device, 
        "Expected both A and B tensors to be on the same device, but got A.device != B.device"
    );
    CHECK_ERROR(
        A->device != C->device, 
        "Expected both A and C tensors to be on the same device, but got A.device != C.device"
    );
    device_t device = C->device;
    if (device == CPU)
        sgemm_cpu(TransA, TransB, M, N, K, alpha, A, offsetA, lda, B, offsetB, ldb, beta, C, offsetC, ldc);
    else if (device == CUDA)
        sgemm_cuda(TransA, TransB, M, N, K, alpha, A, offsetA, lda, B, offsetB, ldb, beta, C, offsetC, ldc);
    else    
        CHECK_ERROR(1, "Given device is not supported.");
}