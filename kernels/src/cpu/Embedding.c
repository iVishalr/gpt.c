#include <cpu/Embedding.h>

void embedding_forward_cpu_kernel(
    const tensor_t *W, 
    const tensor_t *input, 
    tensor_t *output
) {
    int B, T, C;
    B = output->shape[0];
    T = output->shape[1];
    C = output->shape[2];

    float *_W = __builtin_assume_aligned(W->t, 64);
    float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = _out + b * T * C + t * C;
            int ix = (int)_inp[b * T + t];
            float *w_ix = _W + ix * C;
            for (int i = 0; i < C; i++)
                out_bt[i] = w_ix[i];
        }
    }
}

void embedding_backward_cpu_kernel(
    const tensor_t *global_grad,
    const tensor_t *cache,
    tensor_t *dW
) {
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    float *_dW = __builtin_assume_aligned(dW->t, 64);
    float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);
    float *_cache = __builtin_assume_aligned(cache->t, 64);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *global_grad_bt = _global_grad + b * T * C + t * C;
            int ix = (int)_cache[b * T + t];
            float *dW_ix = _dW + ix * C;
            for (int i = 0; i < C; i++) {
                dW_ix[i] += global_grad_bt[i];
            }
        }
    }
}