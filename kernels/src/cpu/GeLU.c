#include <math.h>
#include <omp.h>
#include <cpu/GeLU.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu_kernel(const tensor_t *input, tensor_t *output) {
    const float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);
    const float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;
    for (int i = 0; i < input->length; i++)
    {
        const float x_i = _inp[i];
        float cube = 0.044715f * x_i * x_i * x_i;
        _out[i] = 0.5f * x_i * (1.0f + tanhf(kBeta * (x_i + cube)));
    }
}

void gelu_backward_cpu_kernel(const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout) {
    const float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);
    const float *_cache = __builtin_assume_aligned(cache->t, 64);
    float *_dout = __builtin_assume_aligned(dout->t, 64);
    const float kBeta = M_SQRT2 * M_2_SQRTPI * 0.5f;

    for (int i = 0; i < cache->length; i++)
    {
        const float x = _cache[i];
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
        _dout[i] = _global_grad[i] * (left_derivative + right_derivative);
    }
}
