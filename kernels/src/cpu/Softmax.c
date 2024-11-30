#include <math.h>
#include <omp.h>
#include <cpu/Softmax.h>
#include <cpu/Alloc.h>


void softmax_forward_cpu_kernel(const tensor_t *input, tensor_t *output) {
    int B, T, C;
    B = input->shape[0];
    T = input->shape[1];
    C = input->shape[2];

    const float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    #pragma omp parallel for
    for (int i = 0; i < B * T; i++) {
        const float *input_bt = _inp + i * C;
        float *output_bt = _out + i * C;

        // find maximum for softmax
        float max = -INFINITY;
        for (int j = 0; j < C; j++)
            max = fmaxf(max, input_bt[j]);

        float sum = 0.0f;
        for (int j = 0; j < C; j++) {
            float val = expf(input_bt[j] - max);
            sum += val;
            output_bt[j] = val;
        }
        sum = 1.0f / sum;

        for (int j = 0; j < C; j++)
            output_bt[j] = output_bt[j] * sum;
    }
}


void softmax_backward_cpu_kernel(const tensor_t *global_grad, const tensor_t *cache, tensor_t *dout) {
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    const float *_softmax_output = __builtin_assume_aligned(cache->t, 64);
    const float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);
    float *_dout = __builtin_assume_aligned(dout->t, 64);

    #pragma omp parallel for
    for (int i = 0; i < B * T; i++) {
        const float *output_bt = _softmax_output + i * C;
        const float *grad_output_bt = _global_grad + i * C;
        float *grad_input_bt = _dout + i * C;

        float sum = 0.0f;
        for (int j = 0; j < C; j++)
            sum += grad_output_bt[j] * output_bt[j];

        for (int j = 0; j < C; j++)
            grad_input_bt[j] = (grad_output_bt[j] - sum) * output_bt[j];
    }
}
