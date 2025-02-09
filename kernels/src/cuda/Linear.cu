#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/Tensor.h>
#include <cuda/Linear.h>
#include "utils.h"

C10_LAUNCH_BOUNDS_1(num_threads() * 4)
__global__ void add_bias_cuda_kernel_impl(const float *src, float *dst, const int B, const int T, const int C) {
    const int i = blockIdx.x;
    const int block_size = blockDim.x;
    const int tid = threadIdx.x;

    float *dst_bt = dst + i * C;

    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        dst_bt[j] += src[j];
}

C10_LAUNCH_BOUNDS_1(num_threads() * 4)
__global__ void bias_backward(const float *global_grad, float *db, const int B, const int T, const int C) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if(oc >= C) return;
    float sum = 0.0;
    // grid-wide loop for maximum parallelism
    for (int i = blockIdx.y; i < B * T; i += gridDim.y) {
        sum += global_grad[i * C + oc];
    }
    // and atomically add everything together. atomics within one block are conflict-free!
    atomicAdd(db + oc, sum);
}

#ifdef __cplusplus
extern "C" {
#endif

void linear_forward_cuda_kernel(
    const tensor_t *W, const tensor_t *b,
    const tensor_t *input, tensor_t *output
) {
    const int in_features = W->shape[1];
    const int out_features = W->shape[0];
    const int B = input->shape[0];
    const int T = input->shape[1];

    sgemm_cuda(
        0, 1, B * T, out_features, in_features,
        1.0f, input, 0, in_features,
        W, 0, in_features,
        0.0f, output, 0, out_features
    );

    if (b != NULL) {
        const int block_size = num_threads();
        const int grid_size = B * T;
        cudaStream_t stream = get_cuda_stream();
        add_bias_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(b->t, output->t, B, T, out_features);
        cudaCheck(cudaGetLastError());
    }
}

void linear_backward_cuda_kernel(
    const tensor_t *global_grad, const tensor_t *cache, const tensor_t *W,
    tensor_t *dW, tensor_t *db, tensor_t *dout
) {
    const int in_features = W->shape[1];
    const int out_features = W->shape[0];
    const int B = global_grad->shape[0];
    const int T = global_grad->shape[1];

    sgemm_cuda(
        0, 0, B * T, in_features, out_features,
        1.0f, global_grad, 0, out_features,
        W, 0, in_features,
        0.0f, dout, 0, in_features
    );

    sgemm_cuda(
        1, 0, out_features, in_features, B * T,
        1.0f, global_grad, 0, out_features,
        cache, 0, in_features,
        1.0f, dW, 0, in_features
    );

    if (db != NULL) {
        const int block_size = num_threads();
        const int grid_size = (out_features + block_size - 1) / block_size;
        cudaStream_t stream = get_cuda_stream();
        bias_backward<<<grid_size, block_size, 0, stream>>>(global_grad->t, db->t, B, T, out_features);
        cudaCheck(cudaGetLastError());
    }
}

#ifdef __cplusplus
}
#endif