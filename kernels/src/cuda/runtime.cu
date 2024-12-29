#include <cuda/cuda_common.h>
#include <cuda/runtime.h>

cublasHandle_t cublas_handle;
int initialized = 0;

void setup_cublas_handle() {
    cublasCheck(cublasCreate(&cublas_handle));
    cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    initialized = 1;
}

cublasHandle_t get_cublas_handle() {
    if (initialized == 0) setup_cublas_handle();
    return cublas_handle;
}