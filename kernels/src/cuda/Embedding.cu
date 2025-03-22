#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/Embedding.h>

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void embedding_forward_cuda_kernel_impl(const float *__restrict__ W, const float *__restrict__ input, float *output, const int B, const int T, const int C) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float *output_bt = output + i * C;

    const int curr_idx = (int)input[i];
    const float *w_ix = W + curr_idx * C;

    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        output_bt[j] = w_ix[j];
}


C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void embedding_backward_cuda_kernel_impl(
    const float *__restrict__ global_grad, 
    const float *__restrict__ cache,
    float *dW, 
    const int B, const int T, const int C, const int cache_length
) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float *global_grad_bt = global_grad + i * C;
    const int curr_idx = (int)cache[i % cache_length];
    float *dW_ix = dW + curr_idx * C;

    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        atomicAdd(&dW_ix[j], global_grad_bt[j]);
}

#ifdef __cplusplus
extern "C" {
#endif

void embedding_forward_cuda_kernel(
    const tensor_t *W, 
    const tensor_t *input, 
    tensor_t *output
) { 
    int B, T, C;
    B = output->shape[0];
    T = output->shape[1];
    C = output->shape[2];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;
    cudaStream_t stream = get_cuda_stream();
    embedding_forward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(W->t, input->t, output->t, B, T, C);
    cudaCheck(cudaGetLastError());
}

void embedding_backward_cuda_kernel(const tensor_t *global_grad, const tensor_t *cache, tensor_t *dW) { 
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;
    cudaStream_t stream = get_cuda_stream();
    embedding_backward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(global_grad->t, cache->t, dW->t, B, T, C, cache->length);
    cudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif
