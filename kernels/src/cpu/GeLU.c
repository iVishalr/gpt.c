#include <math.h>
#include <omp.h>
#include <cpu/GeLU.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu_kernel(const tensor_t *input, tensor_t *output) {
    float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    for (int i = 0; i < input->length; i++) {
        float x_i = _inp[i];
        float cube = 0.044715f * x_i * x_i * x_i;
        _out[i] = 0.5f * x_i * (1.0f + tanhf(GELU_SCALING_FACTOR * (x_i + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it
// (https://github.com/karpathy/llm.c/pull/200)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward_cpu_kernel(const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout) {
    float *_global_cache = __builtin_assume_aligned(global_grad->t, 64);
    float *_cache = __builtin_assume_aligned(cache->t, 64);
    float *_dout = __builtin_assume_aligned(dout->t, 64);

    for (int i = 0; i < cache->length; i++) {
        float x_i = _cache[i];
        float cube = 0.044715f * x_i * x_i * x_i;
        float tanh_arg = GELU_SCALING_FACTOR * (x_i + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x_i * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x_i * x_i);
        _dout[i] += local_grad * _global_cache[i];
    }
}
#pragma float_control(pop)
