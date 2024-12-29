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

    const float *_W = __builtin_assume_aligned(W->t, 64);
    const float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    for (int i = 0; i < B * T; i++) {
        float *out_bt = _out + i * C;
        const int ix = (int)_inp[i];
        const float *w_ix = _W + ix * C;
        for (int j = 0; j < C; j++)
            out_bt[j] = w_ix[j];
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

    const int cache_length = cache->length;
    const float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);
    const float *_cache = __builtin_assume_aligned(cache->t, 64);
    float *_dW = __builtin_assume_aligned(dW->t, 64);

    for (int i = 0; i < B * T; i++) {
        const float *global_grad_bt = _global_grad + i * C;
        const int ix = (int)_cache[i % cache_length];
        float *dW_ix = _dW + ix * C;
        for (int j = 0; j < C; j++)
            dW_ix[j] += global_grad_bt[j];
    }
}