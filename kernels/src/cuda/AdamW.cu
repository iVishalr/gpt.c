#include <stdlib.h>
#include <cuda/cuda_common.h>
#include <cuda/AdamW.h>

__device__ inline float lerpf(float start, float end, float weight) {
    return fmaf(weight, end, fmaf(-weight, start, start));
}

C10_LAUNCH_BOUNDS_1(num_threads() * 2)
__global__ void step_adamW_cuda_kernel_impl(
    float **parameters,
    float **gradients,
    float **m,
    float **v,
    const int *lengths,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
) {
    const int i = blockIdx.x;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    float *param = parameters[i];
    float *gradient = gradients[i];
    float *m_t = m[i];
    float *v_t = v[i];
    const int length = lengths[i];

    const float beta1_correction = 1.0f - powf(beta1, step);
    const float beta2_correction = 1.0f - powf(beta2, step);

    for (int j = tid; j < length; j += block_size) {
        param[j] -= lr * weight_decay * param[j];
        float _m = lerpf(gradient[j], m_t[j], beta1);
        float _v = lerpf(gradient[j] * gradient[j], v_t[j], beta2);
        float m_hat = _m / beta1_correction;
        float v_hat = _v / beta2_correction;
        m_t[j] = _m;
        v_t[j] = _v;
        param[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
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

    // assign device pointers
    for (int i = 0; i < n; i++) {
        _parameters[i] = parameters[i]->t;
        _gradients[i] = gradients[i]->t;
        _m[i] = m[i]->t;
        _v[i] = v[i]->t;
        _parameters_sizes[i] = parameters[i]->length;
    }

    // Initialize containers for storing device pointers on device
    float **d_parameters, **d_gradients, **d_m, **d_v;
    int *d_parameters_sizes;
    cudaCheck(cudaMalloc(&d_parameters, n * sizeof(float*)));
    cudaCheck(cudaMalloc(&d_gradients, n * sizeof(float*)));
    cudaCheck(cudaMalloc(&d_m, n * sizeof(float*)));
    cudaCheck(cudaMalloc(&d_v, n * sizeof(float*)));
    cudaCheck(cudaMalloc((void**)&d_parameters_sizes, n * sizeof(int)));

    // Copy over the container data from host to device
    cudaCheck(cudaMemcpy(d_parameters, _parameters, n * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_gradients, _gradients, n * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_m, _m, n * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_v, _v, n * sizeof(float*), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_parameters_sizes, _parameters_sizes, n * sizeof(int), cudaMemcpyHostToDevice));

    // launch kernel
    const int block_size = num_threads() * 2;
    const int grid_size = n;
    step_adamW_cuda_kernel_impl<<<grid_size, block_size>>>(
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
        step
    );
    cudaCheck(cudaGetLastError());

    // free memory
    cudaCheck(cudaFree(d_parameters));
    cudaCheck(cudaFree(d_gradients));
    cudaCheck(cudaFree(d_m));
    cudaCheck(cudaFree(d_v));
    cudaCheck(cudaFree(d_parameters_sizes));
    free(_parameters);
    free(_gradients);
    free(_m);
    free(_v);
    free(_parameters_sizes);
}

void zero_grad_adamW_cuda_kernel(tensor_t **gradients, const int n) {
    for (int i = 0; i < n; i++) {
        tensor_t *gradient = gradients[i];
        float *tensor = gradient->t;
        cudaCheck(cudaMemset(tensor, 0, gradient->length * sizeof(float)));
    }
}

#ifdef __cplusplus
}
#endif
