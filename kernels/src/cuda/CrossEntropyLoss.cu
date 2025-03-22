#include <math.h>
#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/common.cuh>
#include <cuda/Alloc.h>
#include <cuda/CrossEntropyLoss.h>

C10_LAUNCH_BOUNDS_1(num_threads() * 2)
__global__ void cross_entropy_forward_cuda_kernel_impl(
    const float *__restrict__ logits, 
    const float *__restrict__ targets, 
    float *softmax_cache, float *loss_cache, 
    float *output,
    const int B, const int T, const int C
) {
    extern __shared__ float shared[];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const int warp_id = tid / C10_WARP_SIZE;
    const int lane_id = tid % C10_WARP_SIZE;
    const int num_warps = block_size / C10_WARP_SIZE;

    const float *logits_bt = logits + i * C;
    float *softmax_cache_bt = softmax_cache + i * C;

    float max = -INFINITY;
    #pragma unroll
    for(int j = tid; j < C; j += block_size)
        max = fmaxf(max, logits_bt[j]);
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

    float sum = 0.0f;
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        sum += expf(logits_bt[j] - max);
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

    const float log_sum = logf(sum);
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        softmax_cache_bt[j] = logits_bt[j] - max - log_sum;

    __syncthreads();

    int curr_target = (int)targets[i];
    loss_cache[i] = -softmax_cache_bt[curr_target];

    if (warp_id != 0 || lane_id != 0)
        return;
    
    atomicAdd(&output[0], loss_cache[i] / (B * T));
}


C10_LAUNCH_BOUNDS_1(num_threads() * 2)
__global__ void cross_entropy_backward_cuda_kernel_impl(
    const float *__restrict__ global_grad, 
    const float *__restrict__ targets,
    const float *__restrict__ cache,
    float *__restrict__ dout, 
    const int B, const int T, const int C
) {
    extern __shared__ float shared[];
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const int warp_id = tid / C10_WARP_SIZE;
    const int lane_id = tid % C10_WARP_SIZE;
    const int num_warps = block_size / C10_WARP_SIZE;

    const int curr_target = (int)targets[i];
    dout[i * C + curr_target] = -global_grad[i] / (B * T);

    __syncthreads();

    const float *cache_bt = cache + i * C;
    const float *dlog_softmax_bt = dout + i * C;
    float *dout_bt = dout + i * C;

    float sum = 0.0f;
    for (int j = tid; j < C; j += block_size)
        sum += dlog_softmax_bt[j];

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

    for (int j = tid; j < C; j += block_size) {
        dout_bt[j] = dlog_softmax_bt[j] - (expf(cache_bt[j]) * sum);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void cross_entropy_forward_cuda_kernel(
    const tensor_t *logits,
    const tensor_t *targets,
    tensor_t **cache,
    tensor_t *output
) { 
    int B, T, C;
    B = logits->shape[0];
    T = logits->shape[1];
    C = logits->shape[2];

    float *loss_cache = (float*)alloc_cuda(B * T * sizeof(float));

    int block_size = C10_WARP_SIZE;
    int grid_size = B * T;
    size_t shmem_size = (block_size / C10_WARP_SIZE) * sizeof(float);
    cudaStream_t stream = get_cuda_stream();
    cross_entropy_forward_cuda_kernel_impl<<<grid_size, block_size, shmem_size, stream>>>(logits->t, targets->t, cache[0]->t, loss_cache, output->t, B, T, C);
    cudaCheck(cudaGetLastError());
    free_cuda(loss_cache);
}

void cross_entropy_backward_cuda_kernel(
    const tensor_t *global_grad,
    const tensor_t **cache,
    tensor_t *dout
) { 

    const tensor_t *log_softmax_output = cache[0];
    const tensor_t *targets = cache[1];

    int B, T, C;
    B = log_softmax_output->shape[0];
    T = log_softmax_output->shape[1];
    C = log_softmax_output->shape[2];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;
    const size_t shmem_size = (block_size / C10_WARP_SIZE) * sizeof(float);
    cudaStream_t stream = get_cuda_stream();
    cross_entropy_backward_cuda_kernel_impl<<<grid_size, block_size, shmem_size, stream>>>(global_grad->t, targets->t, log_softmax_output->t, dout->t, B, T, C);
    cudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif
