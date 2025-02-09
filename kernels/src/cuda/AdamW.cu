#include <stdlib.h>
#include <cuda/cuda_common.h>
#include <cuda/runtime.cuh>
#include <cuda/Alloc.h>
#include <cuda/Tensor.h>
#include <cuda/AdamW.h>

__device__ inline float lerpf(float start, float end, float weight) {
    return fmaf(weight, end, fmaf(-weight, start, start));
}


C10_LAUNCH_BOUNDS_1(num_threads() * 2)
__global__ void step_adamW_cuda_kernel_impl(
    float **__restrict__ parameters,
    float **__restrict__ gradients,
    float **__restrict__ m,
    float **__restrict__ v,
    const int *__restrict__ lengths,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const float beta1_correction,
    const float beta2_correction
) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    const int j = i * block_size + tid;

    for (int y = 0; y < n; y++) {
        if (j >= lengths[y]) continue;

        const float *gradient = gradients[y];
        float *param = parameters[y];
        float *m_t = m[y];
        float *v_t = v[y];

        const float grad_j = gradient[j];
        float param_j = param[j];
        float _m = m_t[j];
        float _v = v_t[j];

        param_j -= lr * weight_decay * param_j;
        _m = lerpf(grad_j, _m, beta1);
        _v = lerpf(grad_j * grad_j, _v, beta2);
        m_t[j] = _m;
        v_t[j] = _v;
        _m *= beta1_correction;
        _v *= beta2_correction;
        param_j -= lr * _m / (sqrtf(_v) + eps);
        param[j] = param_j;
    }
}


#ifdef __cplusplus
extern "C" {
#endif

void step_adamW_cuda_kernel(
    tensor_t **parameters,
    tensor_t **gradients,
    tensor_t **m,
    tensor_t **v,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
) {
    // Initialize containers for storing device pointers on host
    float **_parameters, **_gradients, **_m, **_v;
    int *_parameters_sizes;
    _parameters = (float**)malloc(n * sizeof(float*));
    _gradients = (float**)malloc(n * sizeof(float*));
    _m = (float**)malloc(n * sizeof(float*));
    _v = (float**)malloc(n * sizeof(float*));
    _parameters_sizes = (int*)malloc(n * sizeof(int));
    int max_length = 0;
    // assign device pointers
    for (int i = 0; i < n; i++) {
        _parameters[i] = parameters[i]->t;
        _gradients[i] = gradients[i]->t;
        _m[i] = m[i]->t;
        _v[i] = v[i]->t;
        _parameters_sizes[i] = parameters[i]->length;
        max_length = parameters[i]->length > max_length ? parameters[i]->length : max_length;
    }

    // Initialize containers for storing device pointers on device
    float **d_parameters = (float**)alloc_cuda(n * sizeof(float*));
    float **d_gradients = (float**)alloc_cuda(n * sizeof(float*));
    float **d_m = (float**)alloc_cuda(n * sizeof(float*));
    float **d_v = (float**)alloc_cuda(n * sizeof(float*));
    int *d_parameters_sizes = (int*)alloc_cuda(n * sizeof(int));

    // Copy over the container data from host to device
    cudaStream_t stream = get_cuda_stream();
    cudaCheck(cudaMemcpyAsync(d_parameters, _parameters, n * sizeof(float*), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_gradients, _gradients, n * sizeof(float*), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_m, _m, n * sizeof(float*), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_v, _v, n * sizeof(float*), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(d_parameters_sizes, _parameters_sizes, n * sizeof(int), cudaMemcpyHostToDevice, stream));

    // sync the stream to complete memcpy
    cudaCheck(cudaStreamSynchronize(stream));

    // launch kernel
    const int block_size = num_threads();
    const int grid_size = (max_length + block_size - 1) / block_size;
    const float beta1_correction = 1.0f / (1.0f - powf(beta1, step));
    const float beta2_correction = 1.0f / (1.0f - powf(beta2, step));

    step_adamW_cuda_kernel_impl<<<grid_size, block_size, 0, stream>>>(
        d_parameters,
        d_gradients,
        d_m,
        d_v,
        d_parameters_sizes,
        n,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        beta1_correction,
        beta2_correction
    );
    cudaCheck(cudaGetLastError());

    // free memory
    free_cuda(d_parameters);
    free_cuda(d_gradients);
    free_cuda(d_m);
    free_cuda(d_v);
    free_cuda(d_parameters_sizes);
    free(_parameters);
    free(_gradients);
    free(_m);
    free(_v);
    free(_parameters_sizes);
}

void zero_grad_adamW_cuda_kernel(tensor_t **gradients, const int n) {
    for (int i = 0; i < n; i++)
        zeros_tensor_data_cuda(gradients[i]);
}

#ifdef __cplusplus
}
#endif
