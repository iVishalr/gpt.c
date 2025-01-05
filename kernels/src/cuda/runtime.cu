#include <cuda/cuda_common.h>
#include <cuda/runtime.h>
#include <cuda/runtime.cuh>
#include "utils.h"

cublasHandle_t cublas_handle;
cudaStream_t cuda_stream;
int runtime_initialized = 0;
int runtime_destroyed = 0;
int runtime_cuda_stream_initialized = 0;
int runtime_cuda_stream_destroyed = 0;
int runtime_cublas_handle_initialized = 0;
int runtime_cublas_handle_destroyed = 0;

void runtime_init_cuda() {
    cudaCheck(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
    cublasCheck(cublasCreate(&cublas_handle));
    // cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
    runtime_cuda_stream_initialized = 1;
    runtime_cublas_handle_initialized = 1;
    runtime_initialized = 1;
}

void runtime_destroy_cuda() {
    cublasCheck(cublasDestroy(cublas_handle));
    cudaCheck(cudaStreamDestroy(cuda_stream));
    runtime_destroyed = 1;
    runtime_cuda_stream_destroyed = 1;
    runtime_cublas_handle_destroyed = 1;
}

cudaStream_t get_cuda_stream() {
    CHECK_ERROR(runtime_cuda_stream_initialized == 0, "cuda_stream has not been created. Please call runtime_cuda_init() for initializing the CUDA runtime environment.");
    CHECK_ERROR(runtime_cuda_stream_destroyed == 1, "cuda_stream has been destoyed. Please call runtime_cuda_init() for initializing the CUDA runtime environment.");
    return cuda_stream;
}

cublasHandle_t get_cublas_handle() {
    CHECK_ERROR(runtime_cublas_handle_initialized == 0, "cublas_handle has not been created. Please call runtime_cuda_init() for initializing the CUDA runtime environment.");
    CHECK_ERROR(runtime_cublas_handle_destroyed == 1, "cublas_handle has been destoyed. Please call runtime_cuda_init() for initializing the CUDA runtime environment.");
    return cublas_handle;
}

void synchronize_cuda() {
    cudaCheck(cudaDeviceSynchronize());
}