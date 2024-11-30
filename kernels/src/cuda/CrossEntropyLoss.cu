#include <math.h>
#include <cuda/common.cuh>
#include <cuda/cuda_common.h>
#include <cuda/CrossEntropyLoss.h>

C10_LAUNCH_BOUNDS_1(num_threads())
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

    const float max_offset = max;

    float sum = 0.0f;
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        sum += expf(logits_bt[j] - max_offset);
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

    const float sum_offset = logf(sum);
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        softmax_cache_bt[j] = logits_bt[j] - max_offset - sum_offset;

    __syncthreads();

    const int curr_target = (int)targets[i];
    loss_cache[i] = -softmax_cache_bt[curr_target];

    __syncthreads();

    float loss = 0.0f;
    const float bt = 1.0f / (B * T);
    for (int j = tid; j < B * T; j += block_size)
        loss += loss_cache[j];
    loss = warp_reduce_sum(loss);

    if (num_warps > 1) {
        if (warp_id == 0)
            shared[lane_id] = 0.0f;
        __syncthreads();
        if (lane_id == 0)
            shared[warp_id] = loss;
        __syncthreads();
        loss = shared[lane_id];
        loss = warp_reduce_sum(loss);
    }

    if (tid == 0)
        output[0] = loss * bt;
}

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void cross_entropy_backward_cuda_kernel_impl(
    const float *__restrict__ global_grad, 
    const float *__restrict__ targets,
    const float *__restrict__ cache,
    float *dlog_softmax, float *dout, 
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
    dlog_softmax[i * C + curr_target] = -global_grad[i] / (B * T);

    __syncthreads();

    const float *cache_bt = cache + i * C;
    const float *dlog_softmax_bt = dlog_softmax + i * C;
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

    float *loss_cache;
    cudaCheck(cudaMalloc((void**)&loss_cache, B * T * sizeof(float)));

    const int block_size = C10_WARP_SIZE * 2;
    const int grid_size = B * T;
    const size_t shmem_size = (block_size / C10_WARP_SIZE) * sizeof(float);
    cross_entropy_forward_cuda_kernel_impl<<<grid_size, block_size, shmem_size>>>(logits->t, targets->t, cache[0]->t, loss_cache, output->t, B, T, C);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaFree(loss_cache));
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

    float *dlog_softmax;
    cudaCheck(cudaMalloc((void**)&dlog_softmax, B * T * C * sizeof(float)));
    cudaCheck(cudaMemset(dlog_softmax, 0, B * T * C * sizeof(float)));

    const int block_size = C10_WARP_SIZE * 2;
    const int grid_size = B * T;
    const size_t shmem_size = (block_size / C10_WARP_SIZE) * sizeof(float);
    cross_entropy_backward_cuda_kernel_impl<<<grid_size, block_size, shmem_size>>>(global_grad->t, targets->t, log_softmax_output->t, dlog_softmax, dout->t, B, T, C);
    cudaCheck(cudaGetLastError());

    cudaCheck(cudaFree(dlog_softmax));
}

#ifdef __cplusplus
}
#endif
