#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/GeLU.h>

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void gelu_forward_cuda_kernel_impl(const float *__restrict__ input, float *output, const int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5;
        const float x_i = input[idx];
        float cube = 0.044715f * x_i * x_i * x_i;
        output[idx] = 0.5f * x_i * (1.0f + tanhf(kBeta * (x_i + cube)));
    }
}

C10_LAUNCH_BOUNDS_1(num_threads())
__global__ void gelu_backward_cuda_kernel_impl(const float* global_grad, const float* input, float* dout, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
        const float x = input[i];
        float x_sq = x * x;
        float x_cube = x_sq * x;
        float inner = kBeta * (x + 0.044715f * x_cube);
        float tanh_inner = tanhf(inner);
        float left = 0.5f * x;
        float right = 1.0f + tanh_inner;

        float left_derivative = 0.5f * right;

        float tanh_derivative = 1.0f - tanh_inner * tanh_inner;
        float inner_derivative = kBeta * (1.0f + 3.0f * 0.044715f * x_sq);
        float right_derivative = left * tanh_derivative * inner_derivative;
        dout[i] = global_grad[i] * (left_derivative + right_derivative);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

void gelu_forward_cuda_kernel(
    const tensor_t *input, tensor_t *output
) {
    const int n = input->length;
    const int block_size = num_threads();
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = get_cuda_stream();
    gelu_forward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(input->t, output->t, n);
    cudaCheck(cudaGetLastError());
}

void gelu_backward_cuda_kernel(
    const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout
) {
    const int n = global_grad->length;
    const int block_size = num_threads();
    const int grid_size = (n + block_size - 1) / block_size;
    cudaStream_t stream = get_cuda_stream();
    gelu_backward_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(global_grad->t, cache->t, dout->t, n);
    cudaCheck(cudaGetLastError());
}

#ifdef __cplusplus
}
#endif
