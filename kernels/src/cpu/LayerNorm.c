#include <math.h>
#include <cpu/Alloc.h>
#include <cpu/LayerNorm.h>

void layer_norm_forward_cpu_kernel(
    const tensor_t *W, 
    const tensor_t *b,
    const tensor_t *input,
    const float eps,
    tensor_t **cache,
    tensor_t *output
) {
    int B, T, in_features;
    B = input->shape[0];
    T = input->shape[1];
    in_features = input->shape[2];

    const float *_inp = __builtin_assume_aligned(input->t, 64);
    const float *_W = __builtin_assume_aligned(W->t, 64);
    const float *_b = b != NULL ? __builtin_assume_aligned(b->t, 64) : NULL;

    float *_out = __builtin_assume_aligned(output->t, 64);
    float *_mean = __builtin_assume_aligned(cache[0]->t, 64);
    float *_rstd = __builtin_assume_aligned(cache[1]->t, 64);

    for(int i = 0; i < B * T; i++) {
        float sum = 0.0f;
        float sum2 = 0.0f;
        
        const float *_inp_i = _inp + i * in_features;
        float *_out_i = _out + i * in_features;

        for (int j = 0; j < in_features; j++) {
            float xi = _inp_i[j];
            sum += xi;
            sum2 += xi * xi;
        }

        sum /= in_features;
        sum2 /= in_features;

        float m = sum;
        float var = sum2 - sum * sum;
        float rstd = 1.0f / sqrtf(var + eps);

        for (int j = 0; j < in_features; j++) {
            float n = rstd * (_inp_i[j] - m);
            float o = _b != NULL ? n * _W[j] + _b[j] : n * _W[j];
            _out_i[j] = o;
        }

        _mean[i] = m;
        _rstd[i] = rstd;
    }
}

void layer_norm_backward_cpu_kernel(
    const tensor_t *global_grad,
    const tensor_t **cache,
    const tensor_t *W,
    tensor_t *dW,
    tensor_t *db,
    tensor_t *dout
) {
    
    int B, T, in_features;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    in_features = global_grad->shape[2];

    tensor_t *mean, *rstd, *x;
    mean = cache[0];
    rstd = cache[1];
    x = cache[2];

    const float *_inp = __builtin_assume_aligned(x->t, 64);
    const float *_mean = __builtin_assume_aligned(mean->t, 64);
    const float *_rstd = __builtin_assume_aligned(rstd->t, 64);
    const float *_W = __builtin_assume_aligned(W->t, 64);
    const float *_global_grad = __builtin_assume_aligned(global_grad->t, 64);

    float *_dW = __builtin_assume_aligned(dW->t, 64);
    float *_db = db != NULL ? __builtin_assume_aligned(db->t, 64) : NULL;
    float *_dout = __builtin_assume_aligned(dout->t, 64);

    for (int i = 0; i < B * T; i++) {
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        
        const float *_inp_i = _inp + i * in_features;
        const float *_global_grad_i = _global_grad + i * in_features;
        float *_dout_i = _dout + i * in_features;

        for(int j = 0; j < in_features; j++) {
            const float norm_i = (_inp_i[j] - _mean[i]) * _rstd[i];
            const float dnorm_i = _W[j] * _global_grad_i[j];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_i;
        }

        dnorm_mean /= in_features;
        dnorm_norm_mean /= in_features;

        for (int j = 0; j < in_features; j++) {
            const float norm_i = (_inp_i[j] - _mean[i]) * _rstd[i];
            const float dnorm_i = _W[j] * _global_grad_i[j];

            if (_db) _db[j] += _global_grad_i[j];
            _dW[j] += norm_i * _global_grad_i[j];

            float dval = 0.0f;
            dval += dnorm_i;
            dval -= dnorm_mean;
            dval -= norm_i * dnorm_norm_mean;
            dval *= _rstd[i];
            _dout_i[j] += dval;
        }
    }
}