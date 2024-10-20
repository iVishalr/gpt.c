#include <cblas.h>
#include <cpu/Linear.h>

void linear_forward_cpu_kernel(const tensor_t *W, const tensor_t *b, const tensor_t *input, tensor_t *output) {
    int B, T, in_features, out_features;
    B = input->shape[0];
    T = input->shape[1];
    in_features = input->shape[2];
    out_features = W->shape[0];

    float *_W = __builtin_assume_aligned(W->t, 64);
    float *_inp = __builtin_assume_aligned(input->t, 64);
    float *_out = __builtin_assume_aligned(output->t, 64);

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        B * T, out_features, in_features,
        1.0f, _inp, in_features,
        _W, in_features,
        0.0f, _out, out_features
    );

    if (b != NULL) {
        float *_b = __builtin_assume_aligned(b->t, 64);
        for (int i = 0; i < B * T; i++) {
            for (int j = 0; j < out_features; j++) {
                _out[i * out_features + j] += _b[j];
            }
        }
    }
}

void linear_backward_cpu_kernel(
    const tensor_t *global_grad, 
    const tensor_t *cache, 
    const tensor_t *W,
    tensor_t *dW, 
    tensor_t *db, 
    tensor_t *dout
) {
    int B, T, in_features, out_features;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    in_features = W->shape[1];
    out_features = global_grad->shape[2];

    float *_W, *_global_grad, *_dW, *_db, *_cache, *_dout;
    _W = __builtin_assume_aligned(W->t, 64);
    _dW = __builtin_assume_aligned(dW->t, 64);
    _dout = __builtin_assume_aligned(dout->t, 64);
    _cache = __builtin_assume_aligned(cache->t, 64);
    _global_grad = __builtin_assume_aligned(global_grad->t, 64);

    // backprop into dx
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        B * T, in_features, out_features,
        1.0f, _global_grad, out_features,
        _W, in_features, 
        1.0f, _dout, in_features
    );

    // backprop into dW
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        out_features, in_features, B * T,
        1.0f, _global_grad, out_features,
        _cache, in_features,
        1.0f, _dW, in_features
    );

    // backprop into db
    if (db != NULL) {
        _db = __builtin_assume_aligned(db->t, 64);
        for (int i = 0; i < B * T; i++) {
            for (int j = 0; j < out_features; j++) {
                _db[j] += _global_grad[i * out_features + j];
            }
        }
    }
}