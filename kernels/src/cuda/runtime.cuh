#include <cuda/cuda_common.h>
#include "utils.h"

extern cublasHandle_t cublas_handle;
extern cudaStream_t cuda_stream;
extern int runtime_initialized;
extern int runtime_destroyed;
extern int runtime_cuda_stream_initialized;
extern int runtime_cuda_stream_destroyed;
extern int runtime_cublas_handle_initialized;
extern int runtime_cublas_handle_destroyed;

cudaStream_t get_cuda_stream();
cublasHandle_t get_cublas_handle();