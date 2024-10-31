#include <math.h>
#include <omp.h>
#include <cpu/CrossEntropyLoss.h>

void cross_entropy_forward_cpu_kernel(
    const tensor_t *probs,
    const tensor_t *targets,
    tensor_t *output
) {
    int B, T, C;
    B = probs->shape[0];
    T = probs->shape[1];
    C = probs->shape[2];

    const float *_probs = __builtin_assume_aligned(probs->t, 64);
    const float *_targets = __builtin_assume_aligned(targets->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *_probs_bt = _probs + b * T * C + t * C;
            int ix = (int)_targets[b * T + t];
            _out[b * T + t] = -logf(_probs_bt[ix]);
        }
    }
}

void cross_entropy_backward_cpu_kernel(
    const tensor_t *global_grad,
    const tensor_t **cache,
    tensor_t *dout
) {

    const tensor_t *probs = cache[0];
    const tensor_t *targets = cache[1];

    int B, T, C;
    B = probs->shape[0];
    T = probs->shape[1];
    C = probs->shape[2];

    const float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);
    const float *_probs = __builtin_assume_aligned(probs->t, 64);
    const float *_targets = __builtin_assume_aligned(targets->t, 64);
    float *_dout = __builtin_assume_aligned(dout->t, 64);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *_dout_bt = _dout + b * T * C + t * C;
            const float *_probs_bt = _probs + b * T * C + t * C;
            const float dloss = _global_grad[b * T + t];
            const int ix = (int)_targets[b * T + t];

            for (int i = 0; i < C; i++) {
                const float p = _probs_bt[i];
                const float indicator = i == ix ? 1.0f : 0.0f;
                _dout_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}