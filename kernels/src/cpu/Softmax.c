#include <math.h>
#include <float.h>
// #include <omp.h>
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
        sum = 1 / sum;

        for (int j = 0; j < C; j++)
            output_bt[j] = output_bt[j] * sum;
    }
}