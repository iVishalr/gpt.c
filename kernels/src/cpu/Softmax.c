#include <math.h>
#include <float.h>
// #include <omp.h>
#include <cpu/Softmax.h>


void softmax_forward_cpu_kernel(const tensor_t *input, tensor_t *output) {
    int B, T, C;
    B = input->shape[0];
    T = input->shape[1];
    C = input->shape[2];

    const float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float *x_bt = _inp + b * T * C + t * C;
            float *out_bt = _out + b * T * C + t * C;
            float maxval = -FLT_MAX;
            for (int i = 0; i < C; i++)
                maxval = x_bt[i] > maxval ? x_bt[i] : maxval;
            
            float sum = 0.0f;
            for (int i = 0; i < C; i++) {
                out_bt[i] = expf(x_bt[i] - maxval);
                sum += out_bt[i];
            }
            for (int i = 0; i < C; i++) {
                out_bt[i] /= sum;
            }
        }
    }
}