#include <cpu/Alloc.h>
#include <cuda/Alloc.h>
#include <cuda/Tensor.h>
#include <cuda/cuda_common.h>
#include <cuda/runtime.h>

#include <common/kutils.h>

#ifdef __cplusplus
extern "C" {
#endif

void move_tensor_to_host_cuda(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(tensor->t == NULL, "Expected *tensor->t to be a float pointer. Got NULL");

    // if tensor is already present on host, return
    if (tensor->device == CPU) return;

    float *device_ptr = tensor->t;
    float *host_ptr = (float*)AllocCheck(aligned_alloc_cpu, tensor->length * sizeof(float), 64);
    cudaCheck(cudaMemcpy(host_ptr, device_ptr, tensor->length * sizeof(float), cudaMemcpyDeviceToHost));

    tensor->t = host_ptr;
    tensor->device = CPU;
    free_cuda(device_ptr);
}

void move_tensor_to_device_cuda(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(tensor->t == NULL, "Expected *tensor->t to be a float pointer. Got NULL");

    // if tensor is already present on device, return
    if (tensor->device == CUDA) return;

    float *host_ptr = tensor->t;
    float *device_ptr;
    const size_t size = tensor->length * sizeof(float);
    cudaCheck(cudaMalloc((void**)&device_ptr, size));
    cudaCheck(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));

    tensor->t = device_ptr;
    tensor->device = CUDA;
    free_cpu(host_ptr);
}

void sgemm_cuda(
    const int TransA, const int TransB, const int M, const int N, const int K,
    const float alpha, const tensor_t *A, const int offsetA, const int lda,
    const tensor_t *B, const int offsetB, const int ldb, 
    const float beta, tensor_t *C, const int offsetC, const int ldc
) {

    float *_A = A->t;
    float *_B = B->t;
    float *_C = C->t;

    cublasOperation_t transa = (TransA == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = (TransB == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Adjust pointers with offsets if provided
    const float *A_ptr = _A + offsetA;
    const float *B_ptr = _B + offsetB;
    float *C_ptr = _C + offsetC;

    cublasHandle_t cublas_handle = get_cublas_handle();

    // // cuBLAS gemm: C = alpha * op(A) * op(B) + beta * C
    cublasCheck(
        cublasSgemm(
            cublas_handle, transb, transa,  // Note swapped order for row-major
            N, M, K,                        // Dimensions
            &alpha,                         // Scalar alpha
            B_ptr, ldb,                     // Matrix B
            A_ptr, lda,                     // Matrix A
            &beta,                          // Scalar beta
            C_ptr, ldc                      // Matrix C
        )
    );
}

#ifdef __cplusplus
}
#endif