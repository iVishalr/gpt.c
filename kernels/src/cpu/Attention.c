#include <omp.h>
#include <math.h>
#include <cblas.h>
#include <string.h>
#include <float.h>
#include <cpu/Alloc.h>
#include <cpu/Attention.h>

void attention_forward_cpu_kernel(
    const tensor_t *input,
    const tensor_t *mask,
    const int n_heads,
    const tensor_t **cache,
    tensor_t *output
) {
    /*
        Computes attention given Q,K,V

        X: (B, T, C * 3)
        q: (B, T, C)
        k: (B, T, C)
        v: (B, T, C)

        q, k, v = X.split(n_embd, dim = -1)
        q = q.view(B, T, n_heads, hs).transpose(1,2) -> (B, n_heads, T, hs)
        k = k.view(B, T, n_heads, hs).transpose(1,2)
        v = v.view(B, T, n_heads, hs).transpose(1,2)

        att: (B, n_heads, T, T) = (q @ k.transpose(-2, -1)) * (1.0f / sqrtf(hs))
        att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        att = Softmax(att, dim = -1)
        out: (B, n_heads, T, hs) = att @ v 
        out = out.transpose(1,2).contiguous().view(B, T, C)
    */

    int B, T, C, C3, hs, mask_row_size;
    B = input->shape[0];
    T = input->shape[1];
    C3 = input->shape[2];
    C = C3 / 3;
    hs = C / n_heads;
    mask_row_size = mask->shape[mask->ndims - 1];
    const float scale = 1.0f / sqrtf(hs);

    const float *x = __builtin_assume_aligned(input->t, 64);
    const float *_mask = __builtin_assume_aligned(mask->t, 64);
    float *q = __builtin_assume_aligned(cache[0]->t, 64);
    float *k = __builtin_assume_aligned(cache[1]->t, 64);
    float *v = __builtin_assume_aligned(cache[2]->t, 64);
    float *preatt = __builtin_assume_aligned(cache[3]->t, 64);
    float *att = __builtin_assume_aligned(cache[4]->t, 64);
    float *out = __builtin_assume_aligned(output->t, 64);

    /*
        The following for loop combines the following operations

        q, k, v = X.split(n_embd, dim = -1)
        q = q.view(B, T, n_heads, hs).transpose(1,2) -> (B, n_heads, T, hs)
        k = k.view(B, T, n_heads, hs).transpose(1,2)
        v = v.view(B, T, n_heads, hs).transpose(1,2)
    */

    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            for (int t = 0; t < T; t++) {
                const float *x_q = x + b * T * C3 + t * C3;
                const float *x_k = x_q + C;
                const float *x_v = x_k + C;

                const float *x_qt = x_q + h * hs;
                const float *x_kt = x_k + h * hs;
                const float *x_vt = x_v + h * hs;

                const int idx =  b * n_heads * T * hs + h * T * hs + t * hs;
                float *qt = q + idx;
                float *kt = k + idx;
                float *vt = v + idx;

                for (int j = 0; j < hs; j++) {
                    qt[j] = x_qt[j];
                    kt[j] = x_kt[j];
                    vt[j] = x_vt[j];
                }
            }
        }
    }

    // att = (q @ k.transpose(-2, -1)) * (1.0/sqrt(hs))
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            const int inp_offset = b * n_heads * T * hs + h * T * hs;
            const int out_offset = b * n_heads * T * T + h * T * T;
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                T, T, hs,
                scale , q + inp_offset, hs,
                k + inp_offset, hs,
                0.0f, att + out_offset, T
            );
        }
    }

    cblas_scopy(B * n_heads * T * T, att, 1, preatt, 1);

    float *_out = (float*)aligned_alloc_cpu(B * n_heads * T * hs * sizeof(float), 64);

    for (int i = 0; i < B * n_heads; i++) {
        for (int j = 0; j < T; j++) {
            float max_val = -FLT_MAX;
            float *att_tt = att + i * T * T + j * T;

            for (int k = 0; k < T; k++) {
                att_tt[k] = _mask[j * mask_row_size + k] == 1.0f ? att_tt[k] : -INFINITY;
                max_val = fmaxf(max_val, att_tt[k]);
            }

            float expsum = 0.0f;
            for (int k = 0; k <= j; k++) {
                float expv = expf(att_tt[k] - max_val);
                expsum += expv;
                att_tt[k] = expv;
            }
            const float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
            for (int k = 0; k < T; k++) {
                if (k <= j)
                    att_tt[k] *= expsum_inv;
                else
                    att_tt[k] = 0.0f;
            }
            
        }

        // compute out = att @ v : att(B, n_heads, T, T) @ v(B, n_heads, T, hs)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            T, hs, T, 
            1.0f, att + i * T * T, T, 
            v + i * T * hs, hs,
            0.0f, _out + i * T * hs, hs
        );
    }

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                memcpy(out + b * T * n_heads * hs + t * n_heads * hs + h * hs, _out + b * n_heads * T * hs + h * T * hs + t * hs, hs * sizeof(float));
            }
        }
    }

    free_cpu(_out);
}

void attention_backward_cpu_kernel(
    const tensor_t *global_grad, 
    const tensor_t **cache,
    const int n_heads,
    tensor_t *dout
) {
    int B, T, C, hs;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];
    hs = C / n_heads;
    const float scale = 1.0f / sqrtf(hs);

    const tensor_t *q, *k, *v, *preatt, *att;
    q = cache[0]; 
    k = cache[1];
    v = cache[2];
    preatt = cache[3];
    att = cache[4];

    float *_global_grad = (float *)aligned_alloc_cpu(B * n_heads * T * hs * sizeof(float), 64);
    float *dq           = (float *)aligned_alloc_cpu(B * n_heads * T * hs * sizeof(float), 64);
    float *dk           = (float *)aligned_alloc_cpu(B * n_heads * T * hs * sizeof(float), 64);
    float *dv           = (float *)aligned_alloc_cpu(B * n_heads * T * hs * sizeof(float), 64);
    float *datt         = (float *)aligned_alloc_cpu(B * n_heads * T * T * sizeof(float), 64);
    float *dpreatt      = (float *)aligned_alloc_cpu(B * n_heads * T * T * sizeof(float), 64);

    for(int i = 0; i < B * n_heads * T * hs; i++) {
        dq[i] = 0.0f;
        dk[i] = 0.0f;
        dv[i] = 0.0f;
    }

    for(int i = 0; i < B * n_heads * T * T; i++) {
        dpreatt[i] = 0.0f;
        datt[i] = 0.0f;
    }

    const float *__global_grad = __builtin_assume_aligned(global_grad->t, 64);
    const float *_q = __builtin_assume_aligned(q->t, 64);
    const float *_k = __builtin_assume_aligned(k->t, 64);
    const float *_v = __builtin_assume_aligned(v->t, 64);
    const float *_att = __builtin_assume_aligned(att->t, 64);
    float *_dout = __builtin_assume_aligned(dout->t, 64);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                memcpy(_global_grad + b * n_heads * T * hs + h * T * hs + t * hs, __global_grad + b * T * n_heads * hs + t * n_heads * hs + h * hs, hs * sizeof(float));
            }
        }
    }

    for(int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            // backprop into q, k
            // dq: (B, n_heads, T, hs)
            // dk: (B, n_heads, T, hs)
            // dpreatt: (B, n_heads, T, T)
            //
            // Forward
            // -------
            // out = att @ v
            //
            // Backward
            // --------
            // datt = global_grad (B, n_heads, T, hs) @ v (B, n_heads, T, hs).T
            // dv = att (B, n_heads, T, T).T @ global_grad (B, n_heads, T, hs)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, 
                T, T, hs,
                1.0f, _global_grad + b * n_heads * T * hs + h * T * hs, hs,
                _v + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, datt + b * n_heads * T * T + h * T * T, T
            );

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans, 
                T, hs, T,
                1.0f, _att + b * n_heads * T * T + h * T * T, T,
                _global_grad + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, dv + b * n_heads * T * hs + h * T * hs, hs
            );

            // backprop into softmax
            const int idx = b * n_heads * T * T + h * T * T;
            for (int t = 0; t < T; t++) {
                const float *att_bth = _att + idx + t * T;
                const float *datt_bth = datt + idx + t * T;
                float *dpreatt_bth = dpreatt + idx + t * T;
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        const float indicator = t2 == t3 ? 1.0f : 0.0f;
                        const float local_grad = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_grad * datt_bth[t2];
                    }
                }
            }

            // backprop into q, k
            // dq: (B, n_heads, T, hs)
            // dk: (B, n_heads, T, hs)
            // dpreatt: (B, n_heads, T, T)
            //
            // Forward
            // -------
            // att = ( q @ k.transpose(-2, -1)) * scale
            //
            // Backward
            // --------
            // dq = dpreatt (B, n_heads, T, T) @ k (B, n_heads, T, hs)
            // dk = dpreatt (B, n_heads, T, T) @ q (B, n_heads, T, hs)

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                T, hs, T,
                scale, dpreatt + b * n_heads * T * T + h * T * T, T,
                _k + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, dq + b * n_heads * T * hs + h * T * hs, hs
            );

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans, 
                T, hs, T,
                scale, dpreatt + b * n_heads * T * T + h * T * T, T,
                _q + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, dk + b * n_heads * T * hs + h * T * hs, hs
            );
        }
    }


    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *dout_q = _dout + b * T * C * 3 + t * C * 3;
            float *dout_k = dout_q + C;
            float *dout_v = dout_k + C;
            for (int h = 0; h < n_heads; h++) {
                const float *_dq = dq + b * n_heads * T * hs + h * T * hs + t * hs;
                const float *_dk = dk + b * n_heads * T * hs + h * T * hs + t * hs;
                const float *_dv = dv + b * n_heads * T * hs + h * T * hs + t * hs;

                float *dout_qh = dout_q + h * hs;
                float *dout_kh = dout_k + h * hs;
                float *dout_vh = dout_v + h * hs;
                #pragma omp simd
                for (int j = 0; j < hs; j++) {
                    dout_qh[j] = _dq[j];
                    dout_kh[j] = _dk[j];
                    dout_vh[j] = _dv[j];
                }
            }
        }
    }

    free_cpu(dq);
    free_cpu(dk);
    free_cpu(dv);
    free_cpu(datt);
    free_cpu(dpreatt);
    free_cpu(_global_grad);
}