#include <math.h>
#include <omp.h>
#include <cpu/AdamW.h>

static inline float lerpf(const float start, const float end, const float weight) {
    return fmaf(weight, end, fmaf(-weight, start, start));
}

void step_adamW_naive_cpu_kernel(
    tensor_t **parameters,
    tensor_t **gradients,
    tensor_t **m,
    tensor_t **v,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
) {
    for (int i = 0; i < n; i++) {
        tensor_t *param, *grad, *m_t, *v_t;
        param = parameters[i];
        grad = gradients[i];
        m_t = m[i];
        v_t = v[i];

        float *_m_t = __builtin_assume_aligned(m_t->t, 64);
        float *_v_t = __builtin_assume_aligned(v_t->t, 64);
        float *_grad_t = __builtin_assume_aligned(grad->t, 64);
        float *_param_t = __builtin_assume_aligned(param->t, 64);

        for (int j = 0; j < grad->length; j++) {
            float _m = beta1 * _m_t[j] + (1.0f - beta1) * _grad_t[j];
            float _v = beta2 * _v_t[j] + (1.0f - beta2) * _grad_t[j] * _grad_t[j];
            float m_hat = _m / (1.0f - powf(beta1, step));
            float v_hat = _v / (1.0f - powf(beta2, step));
            _m_t[j] = _m;
            _v_t[j] = _v;
            _param_t[j] -= lr * (m_hat / (sqrtf(v_hat + eps) + weight_decay * _param_t[j]));
        }
    }
}

void step_adamW_pytorch_cpu_kernel(
    tensor_t **parameters,
    tensor_t **gradients,
    tensor_t **m,
    tensor_t **v,
    const int n,
    const float lr,
    const float beta1,
    const float beta2,
    const float weight_decay,
    const float eps,
    const int step
) {
    const float beta1_correction = 1.0f - powf(beta1, step);
    const float beta2_correction = 1.0f - powf(beta2, step);

    for (int i = 0; i < n; i++) {
        tensor_t *param, *grad, *m_t, *v_t;
        param = parameters[i];
        grad = gradients[i];
        m_t = m[i];
        v_t = v[i];

        float *_m_t = __builtin_assume_aligned(m_t->t, 64);
        float *_v_t = __builtin_assume_aligned(v_t->t, 64);
        float *_grad_t = __builtin_assume_aligned(grad->t, 64);
        float *_param_t = __builtin_assume_aligned(param->t, 64);

        for (int j = 0; j < grad->length; j++) {
            _param_t[j] -= lr * weight_decay * _param_t[j];
            // float _m = lerpf(_grad_t[j], _m_t[j], beta1);                // beta1 * _m_t[j] + (1.0f - beta1) * _grad_t[j];
            // float _v = lerpf(_grad_t[j] * _grad_t[j], _v_t[j], beta2);   // beta2 * _v_t[j] + (1.0f - beta2) * _grad_t[j] * _grad_t[j];
            float _m = beta1 * _m_t[j] + (1.0f - beta1) * _grad_t[j];
            float _v = beta2 * _v_t[j] + (1.0f - beta2) * _grad_t[j] * _grad_t[j];

            float m_hat = _m / beta1_correction;
            float v_hat = _v / beta2_correction;
            _m_t[j] = _m;
            _v_t[j] = _v;
            _param_t[j] -= lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}

void zero_grad_adamW_cpu_kernel(tensor_t **gradients, const int n) {
    for (int i = 0; i < n; i++) {
        tensor_t *grad = gradients[i];
        float *_grad_t = __builtin_assume_aligned(grad->t, 64);
        for (int j = 0; j < grad->length; j++)
            _grad_t[j] = 0.0f;
    }
}