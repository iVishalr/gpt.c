#include <math.h>
#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/common.cuh>
#include <cuda/Softmax.h>

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void softmax_forward_cuda_kernel_impl(const float *__restrict__ input, float *output, const int B, const int T, const int C) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    const int warp_id = tid / C10_WARP_SIZE;
    const int lane_id = tid % C10_WARP_SIZE;
    const int num_warps = block_size / C10_WARP_SIZE;

    extern __shared__ float shared[];

    const float *input_bt = input + i * C;
    float *output_bt = output + i * C;

    float max = -INFINITY;
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        max = fmaxf(max, input_bt[j]);
    max = warp_reduce_max(max);

    if (num_warps > 1) {
        if (warp_id == 0)
            shared[lane_id] = -INFINITY;

        __syncthreads();

        if (lane_id == 0)
            shared[warp_id] = max;
        
        __syncthreads();

        max = shared[lane_id];
        max = warp_reduce_max(max);
    }

    float max_offset = max;

    #pragma unroll
    for (int j = tid; j < C; j += block_size) {
        output_bt[j] = expf(input_bt[j] - max_offset);
    }

    float sum = 0.0f;
    #pragma unroll
    for (int j = tid; j < C; j += block_size) {
        sum += output_bt[j];
    }
    sum = warp_reduce_sum(sum);

    if (num_warps > 1) {
        if (warp_id == 0)
            shared[lane_id] = 0.0f;

        __syncthreads();

        if (lane_id == 0)
            shared[warp_id] = sum;
        
        __syncthreads();

        sum = shared[lane_id];
        sum = warp_reduce_sum(sum);
    }

    const float final_sum = 1.0f / sum;
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        output_bt[j] = output_bt[j] * final_sum;
}

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void softmax_backward_cuda_kernel_impl(
    const float *__restrict__ global_grad, 
    const float *__restrict__ cache,
    float *dout, 
    const int B, const int T, const int C
) {
    extern __shared__ float shared[];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const int warp_id = tid / C10_WARP_SIZE;
    const int lane_id = tid % C10_WARP_SIZE;
    const int num_warps = block_size / C10_WARP_SIZE; 

    const float *global_grad_bt = global_grad + i * C;
    const float *cache_bt = cache + i * C;
    float *dout_bt = dout + i * C;

    float sum = 0.0f;

    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        sum += global_grad_bt[j] * cache_bt[j];
    sum = warp_reduce_sum(sum);

    if (num_warps > 1) {
        if (warp_id == 0)
            shared[lane_id] = 0.0f;
        __syncthreads();
        if (lane_id == 0)
            shared[warp_id] = sum;
        __syncthreads();
        sum = shared[lane_id];
        sum = warp_reduce_sum(sum);
    }

    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        dout_bt[j] = (global_grad_bt[j] - sum) * cache_bt[j];
}

#ifdef __cplusplus
extern "C" {
#endif

void softmax_forward_cuda_kernel(const tensor_t *input, tensor_t *output) { 
    int B, T, C;
    B = input->shape[0];
    T = input->shape[1];
    C = input->shape[2];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;
    const size_t shmem_size = (block_size / C10_WARP_SIZE) * sizeof(float);
    cudaStream_t stream = get_cuda_stream();
    softmax_forward_cuda_kernel_impl<<<grid_size, block_size, shmem_size, stream>>>(input->t, output->t, B, T, C);
    cudaCheck(cudaGetLastError());
}

void softmax_backward_cuda_kernel(const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout) { 
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;
    const size_t shmem_size = (block_size / C10_WARP_SIZE) * sizeof(float);
    cudaStream_t stream = get_cuda_stream();
    softmax_backward_cuda_kernel_impl<<<grid_size, block_size, shmem_size, stream>>>(global_grad->t, cache->t, dout->t, B, T, C);
    cudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif
