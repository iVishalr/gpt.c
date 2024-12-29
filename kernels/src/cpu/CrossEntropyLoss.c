#include <math.h>
#include <float.h>
#include <omp.h>
#include <stdlib.h>
#include <cpu/CrossEntropyLoss.h>

void cross_entropy_forward_cpu_kernel(
    const tensor_t *logits,
    const tensor_t *targets,
    tensor_t **cache,
    tensor_t *output
) {
    int B, T, C;
    B = logits->shape[0];
    T = logits->shape[1];
    C = logits->shape[2];

    const float *_logits = __builtin_assume_aligned(logits->t, 64);
    const float *_targets = __builtin_assume_aligned(targets->t, 64);
    float *_out = (float*)aligned_alloc(64, B * T * sizeof(float));
    
    // stores log_softmax output for backward pass
    float *tmp_logits = __builtin_assume_aligned(cache[0]->t, 64);

    #pragma omp parallel for
    for (int i = 0; i < B * T; i++) {   
        const float *logits_bt = _logits + i * C;
        float *tmp_logits_bt = tmp_logits + i * C;

        // find maximum for softmax
        float max = -INFINITY;
        for (int j = 0; j < C; j++)
            max = fmaxf(max, logits_bt[j]);

        // calculate log(softmax(logits)) which is just logits - max
        float sum = 0.0f;
        for (int j = 0; j < C; j++)
            sum += expf(logits_bt[j] - max);
        sum = logf(sum);

        for (int j = 0; j < C; j++)
            tmp_logits_bt[j] = logits_bt[j] - max - sum;

        const int ix = (int)_targets[i];
        _out[i] = -tmp_logits_bt[ix];
    }

    float loss = 0.0f;
    for (int i = 0; i < B * T; i++)
        loss += _out[i];

    output->t[0] = loss / (B * T);
    free(_out);
}

void cross_entropy_backward_cpu_kernel(
    const tensor_t *global_grad,
    const tensor_t **cache,
    tensor_t *dout
) {

    const tensor_t *log_softmax_output = cache[0];
    const tensor_t *targets = cache[1];

    int B, T, C;
    B = log_softmax_output->shape[0];
    T = log_softmax_output->shape[1];
    C = log_softmax_output->shape[2];

    // nll_backward
    float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);
    float *_targets = __builtin_assume_aligned(targets->t, 64);
    float *dlog_softmax = __builtin_assume_aligned(dout->t, 64);

    const int bt = B * T;
    for (int i = 0; i < B * T; i++) {
        const int curr_target = (int)_targets[i];
        dlog_softmax[i * C + curr_target] = -_global_grad[i] / bt;
    }

    // log_softmax backward
    float *_out = __builtin_assume_aligned(log_softmax_output->t, 64);
    float *_dout = __builtin_assume_aligned(dout->t, 64);

    #pragma omp parallel for
    for (int i = 0; i < B * T; i++) {
        const float *output_bt = _out + i * C;
        const float *grad_output_bt = dlog_softmax + i * C;
        float *grad_input_bt = _dout + i * C;

        float sum = 0.0f;
        for (int j = 0; j < C; j++)
            sum += grad_output_bt[j];

        for (int j = 0; j < C; j++)
            grad_input_bt[j] = grad_output_bt[j] - (expf(output_bt[j]) * sum);
    }
}