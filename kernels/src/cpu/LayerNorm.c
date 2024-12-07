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

    /*
    Implementation taken from:
    https://github.com/pytorch/pytorch/blob/b6a64b64de87cddd16c528215acae73502ca4611/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L28
    */

    const float scale = 1.0f / in_features;
    for(int i = 0; i < B * T; i++) {
        float mean_var[2] = {0.0f, 0.0f};
        const float *input_bt = _inp + i * in_features;
        float *output_bt = _out + i * in_features;

        for (int j = 0; j < in_features; j++) {
            float xi = input_bt[j];
            mean_var[0] += xi;
            mean_var[1] += xi * xi;
        }

        const float mean_val = mean_var[0] * scale;
        const float __b = -mean_val;
        const float var = mean_var[1] * scale - mean_val * mean_val;
        const float rstd_val = 1.0f / sqrtf(var + eps);

        if (_b)
            for (int j = 0; j < in_features; j++)
                output_bt[j] = (input_bt[j] + __b) * rstd_val * _W[j] + _b[j];
        else
            for (int j = 0; j < in_features; j++)
                output_bt[j] = (input_bt[j] + __b) * rstd_val * _W[j];
        _mean[i] = mean_val;
        _rstd[i] = rstd_val;
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

    const tensor_t *mean, *rstd, *x;
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

    /*
    Implementation taken from:
    https://github.com/pytorch/pytorch/blob/b6a64b64de87cddd16c528215acae73502ca4611/aten/src/ATen/native/cpu/layer_norm_kernel.cpp#L185

    This implementation is ~2X faster compared to:
    https://github.com/karpathy/llm.c/blob/7ecd8906afe6ed7a2b2cdb731c042f26d525b820/train_gpt2.c#L120
    */

    const float scale = 1.0f / in_features;
    for (int i = 0; i < B * T; i++) {
        const float *input_bt = _inp + i * in_features;
        const float *global_grad_bt = _global_grad + i * in_features;
        float *dout_bt = _dout + i * in_features;
        {
        const float a = _rstd[i];
        const float b = -a * _mean[i];
        for (int j = 0; j < in_features; j++)
            _dW[j] += (a * input_bt[j] + b) * global_grad_bt[j];
        }
        if (_db)
            for (int j = 0; j < in_features; j++)
                _db[j] += global_grad_bt[j];

        float ds_acc = 0.0f;
        float db_acc = 0.0f;
        for (int j = 0; j < in_features; j++) {
            ds_acc += global_grad_bt[j] * input_bt[j] * _W[j];
            db_acc += global_grad_bt[j] * _W[j];
        }
        {
        const float a = _rstd[i];
        const float b = (db_acc * _mean[i] - ds_acc) * a * a * a * scale;
        const float c = -b * _mean[i] - db_acc * a * scale;
        for (int j = 0; j < in_features; j++)
            dout_bt[j] += a * global_grad_bt[j] * _W[j] + b * input_bt[j] + c;
        }
    }
}