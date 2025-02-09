#include <math.h>
#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/common.cuh>
#include <cuda/Alloc.h>
#include <cuda/Tensor.h>
#include <cuda/Softmax.h>
#include <cuda/Attention.h>
#include "utils.h"


C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void apply_mask_cuda_kernel_impl(const float *mask, float *input, const int B, const int T, const int n_heads, const int ldmask) {
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int i = blockIdx.x;

    float *att_tt = input + i * T * T;
    for (int j = tid; j < T; j += block_size) {
        float *att_j = att_tt + j * T;
        const float *mask_j = mask + j * ldmask;
        for (int k = 0; k < T; k++) {
            att_j[k] = mask_j[k] == 1.0f ? att_j[k] : -INFINITY;
        }
    }
}


C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void permute_forward_cuda_kernel_impl(const float *input, float *q, float *k, float *v, const int B, const int T, const int n_heads, const int hs) {
    // okay so now, this kernel wants Q,K,V to all be of shape (B, n_heads, T, hs)
    // but instead, we have a single tensor QKV (input) of shape (B, T, 3, n_heads, hs)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Q[b][nh_][n][d_] = input[b][n][0][nh_][d_]
    if (idx < B * n_heads * T * hs) {
        int b = idx / (n_heads * T * hs);
        int rest = idx % (n_heads * T * hs);
        int nh_ = rest / (T * hs);
        rest = rest % (T * hs);
        int n = rest / hs;
        int d_ = rest % hs;

        int inp_idx = \
            (b * T * 3 * n_heads * hs)
            +   (n * 3 * n_heads * hs)
            +       (0 * n_heads * hs)
            +          (nh_ * hs)
            +                d_;

        q[idx] = __ldcs(&input[inp_idx]);
        k[idx] = __ldcs(&input[inp_idx + n_heads * hs]);
        v[idx] = __ldcs(&input[inp_idx + 2 * (n_heads * hs)]);
    }
}


C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unpermute_forward_cuda_kernel_impl(const float *input, float *output, const int B, const int T, const int n_heads, const int hs) {
   // out has shape (B, nh, T, hs) but we need to unpermute it to (B, T, nh, hs)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // out[b][n][nh_][d_] <- input[b][nh_][n][d_]
    if (idx < B * n_heads * T * hs) {
        int b = idx / (n_heads * T * hs);
        int rest = idx % (n_heads * T * hs);
        int nh_ = rest / (T * hs);
        rest = rest % (T * hs);
        int n = rest / hs;
        int d_ = rest % hs;

        int other_idx = (b * n_heads * T * hs) + (n * n_heads * hs) + (nh_ * hs) + d_;
        output[other_idx] = __ldcs(&input[idx]);
    }
}


C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void permute_backward_cuda_kernel_impl(const float *dq, const float *dk, const float *dv, float *dout, const int B, const int T, const int n_heads, const int hs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * n_heads * T * hs) {
        int b = idx / (n_heads * T * hs);
        int rest = idx % (n_heads * T * hs);
        int nh_ = rest / (T * hs);
        rest = rest % (T * hs);
        int n = rest / hs;
        int d_ = rest % hs;

        int inp_idx = (b * T * 3 * n_heads * hs) + (n * 3 * n_heads * hs) + (0 * n_heads * hs) + (nh_ * hs) + d_;
        dout[inp_idx] += dq[idx];
        dout[inp_idx + n_heads * hs] += dk[idx];
        dout[inp_idx + 2 * (n_heads * hs)] += dv[idx];
    }
}


C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void unpermute_backward_cuda_kernel_impl(const float *global_grad, float *dout, int B, int T, int n_heads, int hs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * n_heads * T * hs) {
        int b = idx / (n_heads * T * hs);
        int rest = idx % (n_heads * T * hs);
        int nh_ = rest / (T * hs);
        rest = rest % (T * hs);
        int n = rest / hs;
        int d_ = rest % hs;

        int other_idx = (b * n_heads * T * hs) + (n * n_heads * hs) + (nh_ * hs) + d_;
        dout[idx] += global_grad[other_idx];
    }
}


void apply_mask_cuda_kernel(const float *mask, float *input, const int B, const int T, const int n_heads, const int ldmask) {
    const int block_size = num_threads();
    const int grid_size = B * n_heads;
    cudaStream_t stream = get_cuda_stream();
    apply_mask_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(mask, input, B, T, n_heads, ldmask);
    cudaCheck(cudaGetLastError());
}


void permute_forward_cuda_kernel(const float *input, float *q, float *k, float *v, const int B, const int T, const int C, const int n_heads) {
    const int hs = C / n_heads;
    const int block_size = num_threads();
    const int total_threads = B * T * n_heads * hs;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    cudaStream_t stream = get_cuda_stream();
    permute_forward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(input, q, k, v, B, T, n_heads, hs);
    cudaCheck(cudaGetLastError());
}


void permute_backward_cuda_kernel(const float *dq, const float *dk, const float *dv, float *dout, const int B, const int T, const int C, const int n_heads) {
    const int hs = C / n_heads;
    const int block_size = num_threads();
    const int total_threads = B * T * n_heads * hs;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    cudaStream_t stream = get_cuda_stream();
    permute_backward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(dq, dk, dv, dout, B, T, n_heads, hs);
    cudaCheck(cudaGetLastError());
}


void unpermute_forward_cuda_kernel(const float *input, float *output, const int B, const int T, const int C, const int n_heads) {
    const int block_size = num_threads();
    const int total_threads = B * T * C;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    cudaStream_t stream = get_cuda_stream();
    unpermute_forward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(input, output, B, T, n_heads, C / n_heads);
    cudaCheck(cudaGetLastError());
}


void unpermute_backward_cuda_kernel(const float *global_grad, float *dout, const int B, const int T, const int C, const int n_heads) {
    const int hs = C / n_heads;
    const int block_size = num_threads();
    const int total_threads = B * T * C;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    cudaStream_t stream = get_cuda_stream();
    unpermute_backward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(global_grad, dout, B, T, n_heads, hs);
    cudaCheck(cudaGetLastError());
}


#ifdef __cplusplus
extern "C" {
#endif

void attention_forward_cuda_kernel(
    const tensor_t *input,
    const tensor_t *mask,
    const int n_heads,
    tensor_t **cache,
    tensor_t *output
) {
    int B, T, C3, C, hs, mask_row_size;
    B = input->shape[0];
    T = input->shape[1];
    C3 = input->shape[2];
    C = C3 / 3;
    hs = C / n_heads;
    mask_row_size = mask->shape[mask->ndims - 1];

    const float scale = 1.0f / sqrtf(hs);
    tensor_t *k, *q, *v, *att;
    q = cache[0];
    k = cache[1];
    v = cache[2];
    att = cache[3];

    // permute 
    permute_forward_cuda_kernel(input->t, q->t, k->t, v->t, B, T, C, n_heads);
    
    // att = (q @ k.transpose(-2, -1)) * (1.0/sqrt(hs))
    sgemm_strided_batched_cuda(
        0, 1, 
        T, T, hs, 
        scale, 
        q, hs, T * hs, 
        k, hs, T * hs, 
        0.0f, 
        att, T, T * T, 
        B * n_heads
    );

    // apply mask
    apply_mask_cuda_kernel(mask->t, att->t, B, T, n_heads, mask_row_size);

    att->shape[1] = n_heads * T;
    softmax_forward_cuda_kernel(att, att);
    att->shape[1] = n_heads;

    // out = att @ v
    float *out = (float*)alloc_cuda(B * n_heads * T * hs * sizeof(float));
    float *tmp = output->t;
    output->t = out;
    sgemm_strided_batched_cuda(
        0, 0,
        T, hs, T,
        1.0f, 
        att, T, T * T,
        v, hs, T * hs,
        0.0f,
        output, hs, T * hs,
        B * n_heads
    );
    output->t = tmp;

    unpermute_forward_cuda_kernel(out, output->t, B, T, C, n_heads);
    free_cuda(out);
}


void attention_backward_cuda_kernel(
    const tensor_t *global_grad, 
    tensor_t **cache,
    const int n_heads,
    tensor_t *dout
) {
    int B, T, C, hs;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];
    hs = C / n_heads;

    const float scale = 1.0f / sqrtf(hs);
    
    tensor_t *k, *q, *v, *att;
    q = cache[0];
    k = cache[1];
    v = cache[2];
    att = cache[3];

    tensor_t *dq, *dk, *dv, *datt, *_global_grad;
    _global_grad = create_tensor(global_grad->shape, global_grad->ndims, CUDA);
    dv           = create_tensor(v->shape, v->ndims, CUDA);
    datt         = create_tensor(att->shape, att->ndims, CUDA);

    // Reuse tensors to save on memory
    // _global_grad = dout;
    dq = _global_grad;
    dk = v;

    unpermute_backward_cuda_kernel(global_grad->t, _global_grad->t, B, T, C, n_heads);

    // datt = global_grad (B, n_heads, T, hs) @ v (B, n_heads, T, hs).T
    sgemm_strided_batched_cuda(
        0, 1, 
        T, T, hs,
        1.0f, 
        _global_grad, hs, T * hs,
        v, hs, T * hs,
        0.0f, 
        datt, T, T * T,
        B * n_heads
    );

    // dv = att (B, n_heads, T, T).T @ global_grad (B, n_heads, T, hs)
    sgemm_strided_batched_cuda(
        1, 0, 
        T, hs, T,
        1.0f, 
        att, T, T * T,
        _global_grad, hs, T * hs,
        0.0f, 
        dv, hs, T * hs,
        B * n_heads
    );

    datt->shape[1] = n_heads * T;
    softmax_backward_cuda_kernel(datt, att, datt);
    datt->shape[1] = n_heads;

    // dq = dpreatt (B, n_heads, T, T) @ k (B, n_heads, T, hs)
    sgemm_strided_batched_cuda(
        0, 0, 
        T, hs, T,
        scale, 
        datt, T, T * T,
        k, hs, T * hs,
        0.0f, 
        dq, hs, T * hs,
        B * n_heads
    );

    // dk = dpreatt (B, n_heads, T, T) @ q (B, n_heads, T, hs)
    sgemm_strided_batched_cuda(
        1, 0, 
        T, hs, T,
        scale, 
        datt, T, T * T,
        q, hs, T * hs,
        0.0f, 
        dk, hs, T * hs,
        B * n_heads
    );

    permute_backward_cuda_kernel(dq->t, dk->t, dv->t, dout->t, B, T, C, n_heads);

    free_tensor(_global_grad);
    free_tensor(dv);
    free_tensor(datt);
}

#ifdef __cplusplus
}
#endif
