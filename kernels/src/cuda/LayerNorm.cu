#include <cuda/LayerNorm.h>
#include <cuda/common.cuh>
#include <cuda/cuda_common.h>

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void layer_norm_forward_cuda_kernel_impl(
    const float *input, const float *W, const float *b, const float eps, 
    float *mean, float *rstd, float *output, 
    const int B, const int T, const int C
) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float *input_bt = input + i * C;
    float *output_bt = output + i * C;

    float2 mean_var = make_float2(0.0f, 0.0f);

    for (int j = tid; j < C; j += block_size) {
        float xi = input_bt[j];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    mean_var = warp_reduce_sum(mean_var);
    if (block_size > C10_WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / C10_WARP_SIZE;
        int lane_id = threadIdx.x % C10_WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float scale = 1.0f / C;
    const float mean_val = mean_var.x * scale;
    const float _b = -mean_val;
    const float var = mean_var.y * scale - mean_val * mean_val;
    const float rstd_val = 1.0f / sqrtf(var + eps);

    if (b) {
        #pragma unroll
        for (int j = tid; j < C; j += block_size)
            output_bt[j] = (input_bt[j] + _b) * rstd_val * W[j] + b[j];
    }
    else {
        #pragma unroll
        for (int j = tid; j < C; j += block_size)
            output_bt[j] = (input_bt[j] + _b) * rstd_val * W[j];
    }

    mean[i] = mean_val;
    rstd[i] = rstd_val;
}

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void layer_norm_backward_cuda_kernel_impl(
    const float *global_grad, const float *input, const float *mean, const float *rstd, const float *W, 
    float *dW, float *db, float *dout, 
    const int B, const int T, const int C
) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    const float scale = 1.0f / C;
    const float *input_bt = input + i * C;
    const float *global_grad_bt = global_grad + i * C;
    float *dout_bt = dout + i * C;

    const float a = rstd[i];
    const float b = -a * mean[i];

    for (int j = tid; j < C; j += block_size)
        atomicAdd(&dW[j], global_grad_bt[j] * (a * input_bt[j] + b));

    if (db)
        for (int j = tid; j < C; j += block_size)
            atomicAdd(&db[j], global_grad_bt[j]);
    
    float2 d_acc = make_float2(0.0f, 0.0f);
    for (int j = tid; j < C; j += block_size) {
        d_acc.x += global_grad_bt[j] * input_bt[j] * W[j];
        d_acc.y += global_grad_bt[j] * W[j];
    }
    d_acc = warp_reduce_sum(d_acc);
    if (block_size > C10_WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / C10_WARP_SIZE;
        int lane_id = threadIdx.x % C10_WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = d_acc;
        }
        __syncthreads();
        d_acc = s_sum[lane_id];
        d_acc = warp_reduce_sum(d_acc);
    }

    const float _b = (d_acc.y * mean[i] - d_acc.x) * a * a * a * scale;
    const float c = -_b * mean[i] - d_acc.y * a * scale;
    #pragma unroll
    for (int j = tid; j < C; j += block_size)
        dout_bt[j] += a * global_grad_bt[j] * W[j] + _b * input_bt[j] + c;
}

#ifdef __cplusplus
extern "C" {
#endif

void layer_norm_forward_cuda_kernel(
    const tensor_t *W, 
    const tensor_t *b,
    const tensor_t *input,
    const float eps,
    tensor_t **cache,
    tensor_t *output
) {
    int B, T, C;
    B = input->shape[0];
    T = input->shape[1];
    C = input->shape[2];

    tensor_t *mean = cache[0];
    tensor_t *rstd = cache[1];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;
    layer_norm_forward_cuda_kernel_impl<<<grid_size, block_size>>>(input->t, W->t, b->t, eps, mean->t, rstd->t, output->t, B, T, C);
    cudaCheck(cudaGetLastError());
}

void layer_norm_backward_cuda_kernel(
    const tensor_t *global_grad,
    const tensor_t **cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
) {
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    const tensor_t *mean, *rstd, *input;
    mean = cache[0];
    rstd = cache[1];
    input = cache[2];

    const int block_size = C10_WARP_SIZE;
    const int grid_size = B * T;

    float *_db = db ? db->t : NULL;
    layer_norm_backward_cuda_kernel_impl<<<grid_size, block_size>>>(global_grad->t, input->t, mean->t, rstd->t, W->t, dW->t, _db, dout->t, B, T, C);
    cudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif