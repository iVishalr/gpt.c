#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "utils.h"
#include "attention.h"

tensor_t *forward_attention(attention_t *attn, tensor_t *x) {

    if (attn == NULL) {
        printf("Expected required arugment *attn to be of type attention_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    /*
        Explanation
        -----------

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

        Layer Caches
        ------------
        When computing forward pass, we first copy data for Q, K, V
        Q, K, V needs to be cached so that they can be accessed during backprop.

        We use Q & K to compute the attention scores and apply masking operation on
        the attention scores. This output is now called "preatt". We should cache this
        as well as this becomes the input to the softmax activation and will be accessed
        during softmax backprop.

        After computing softmax on attention scores, the attention scores are again cached
        as this becomes input to "out = att @ v", and will be accessed during backprop.
    */

    int B, T, C, C3, n_heads, hs, buffer_row_size;
    B = x->shape[0];
    T = x->shape[1];
    C3 = x->shape[2];
    C = C3 / 3;
    n_heads = attn->n_heads;
    hs = C / n_heads;
    buffer_row_size = attn->buffer->shape[attn->buffer->ndims - 1];
    float scale = 1.0f / sqrtf(hs);

    tensor_t *q, *k, *v; 
    int qkv_transpose_shape[4] = {B, n_heads, T, hs};
    q = create_tensor(qkv_transpose_shape, 4);
    k = create_tensor(qkv_transpose_shape, 4);
    v = create_tensor(qkv_transpose_shape, 4);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *x_q = x->t + b * T * C3 + t * C3;
            float *x_k = x_q + C;
            float *x_v = x_k + C;
            for (int h = 0; h < n_heads; h++) {
                memcpy(q->t + b * n_heads * T * hs + h * T * hs + t * hs, x_q + h * hs, hs * sizeof(float));
                memcpy(k->t + b * n_heads * T * hs + h * T * hs + t * hs, x_k + h * hs, hs * sizeof(float));
                memcpy(v->t + b * n_heads * T * hs + h * T * hs + t * hs, x_v + h * hs, hs * sizeof(float));
            }
        }
    }

    int att_shape[4] = {B, n_heads, T, T};
    tensor_t *att = create_tensor(att_shape, 4);

    // att = (q @ k.transpose(-2, -1)) * (1.0/sqrt(hs))
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < n_heads; h++) {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                T, T, hs,
                scale , q->t + b * n_heads * T * hs + h * T * hs, hs,
                k->t + b * n_heads * T * hs + h * T * hs, hs,
                0.0f, att->t + b * n_heads * T * T + h * T * T, T
            );
        }
    }

    // cache current att scores
    tensor_t *preatt = create_tensor(att->shape, att->ndims);
    tensor_copy(preatt, att);

    int out_shape[3] = {B, T, C};
    tensor_t *out, *_out; 
    out = create_tensor(out_shape, 3);
    _out = create_tensor(out_shape, 3);

    for (int i = 0; i < B * n_heads; i++) {
        for (int j = 0; j < T; j++) {
            float max_val = -INFINITY;
            float *att_tt = att->t + i * T * T + j * T;

            for (int k = 0; k < T; k++) {
                att_tt[k] = attn->buffer->t[j * buffer_row_size + k] == 1.0f ? att_tt[k] : -INFINITY;
                if (att_tt[k] > max_val) 
                    max_val = att_tt[k];
            }

            float expsum = 0.0f;
            for (int k = 0; k <= j; k++) {
                float expv = expf(att_tt[k] - max_val);
                expsum += expv;
                att_tt[k] = expv;
            }
            float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
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
            1.0f, att->t + i * T * T, T, 
            v->t + i * T * hs, hs,
            0.0f, _out->t + i * T * hs, hs
        );
    }

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                memcpy(out->t + b * T * n_heads * hs + t * n_heads * hs + h * hs, _out->t + b * n_heads * T * hs + h * T * hs + t * hs, hs * sizeof(float));
            }
        }
    }
    
    attn->cache[0] = q;
    attn->cache[1] = k;
    attn->cache[2] = v;
    attn->cache[3] = preatt;
    attn->cache[4] = att;
    
    free_tensor(x);
    free_tensor(_out);
    
    return out;
}

tensor_t *backward_attention(attention_t *attn, tensor_t *global_grad) {

    if (attn == NULL) {
        printf("Expected required arugment *attn to be of type attention_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    int B, T, C, n_heads, hs;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];
    n_heads = attn->n_heads;
    hs = C / n_heads;
    float scale = 1.0f / sqrtf(hs);

    int datt_shape[4] = {B, n_heads, T, T};
    tensor_t *preatt, *dpreatt, *att, *datt, *q, *dq, *k, *dk, *v, *dv;

    q = attn->cache[0];
    k = attn->cache[1];
    v = attn->cache[2];
    preatt = attn->cache[3];
    att = attn->cache[4];

    dq = zeros(q->shape, q->ndims);
    dk = zeros(k->shape, k->ndims);
    dv = zeros(v->shape, v->ndims);
    dpreatt = zeros(datt_shape, 4);
    datt = zeros(datt_shape, 4);

    int global_grad_T_shape[4] = {B, n_heads, T, hs};
    tensor_t *global_grad_T = create_tensor(global_grad_T_shape, 4);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < n_heads; h++) {
                memcpy(global_grad_T->t + b * n_heads * T * hs + h * T * hs + t * hs, global_grad->t + b * T * n_heads * hs + t * n_heads * hs + h * hs, hs * sizeof(float));
            }
        }
    }

    free_tensor(global_grad);
    global_grad = global_grad_T;
    global_grad_T = NULL;

    for (int b = 0; b < B; b++) {
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
                1.0f, global_grad->t + b * n_heads * T * hs + h * T * hs, hs,
                v->t + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, datt->t + b * n_heads * T * T + h * T * T, T
            );

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                T, hs, T,
                1.0f, att->t + b * n_heads * T * T + h * T * T, T,
                global_grad->t + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, dv->t + b * n_heads * T * hs + h * T * hs, hs 
            );

            // backprop into softmax

            for (int t = 0; t < T; t++) {
                float *att_bth = att->t + b * n_heads * T * T + h * T * T + t * T;
                float *datt_bth = datt->t + b * n_heads * T * T + h * T * T + t * T;
                float *dpreatt_bth = dpreatt->t + b * n_heads * T * T + h * T * T + t * T;
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_grad = att_bth[t2] * (indicator - att_bth[t3]);
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
                scale, dpreatt->t + b * n_heads * T * T + h * T * T, T,
                k->t + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, dq->t + b * n_heads * T * hs + h * T * hs, hs
            );

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                T, hs, T,
                scale, dpreatt->t + b * n_heads * T * T + h * T * T, T,
                q->t + b * n_heads * T * hs + h * T * hs, hs,
                1.0f, dk->t + b * n_heads * T * hs + h * T * hs, hs
            );
        }
    }

    free_tensor(global_grad);
    free_tensor(datt);
    free_tensor(dpreatt);

    // accumulate all gradients to dout and free the tensors
    int dout_shape[3] = {B, T, C * 3};
    tensor_t *dout = create_tensor(dout_shape, 3);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *dout_q = dout->t + b * T * C * 3 + t * C * 3;
            float *dout_k = dout_q + C;
            float *dout_v = dout_k + C;
            for (int h = 0; h < n_heads; h++) {
                memcpy(dout_q + h * hs, dq->t + b * n_heads * T * hs + h * T * hs + t * hs, hs * sizeof(float));
                memcpy(dout_k + h * hs, dk->t + b * n_heads * T * hs + h * T * hs + t * hs, hs * sizeof(float));
                memcpy(dout_v + h * hs, dv->t + b * n_heads * T * hs + h * T * hs + t * hs, hs * sizeof(float));
            }
        }
    }

    free_tensor(q);
    free_tensor(k);
    free_tensor(v);
    free_tensor(preatt);
    free_tensor(att);
    free_tensor(dq);
    free_tensor(dk);
    free_tensor(dv);
    attn->cache[0] = NULL;
    attn->cache[1] = NULL;
    attn->cache[2] = NULL;
    attn->cache[3] = NULL;
    attn->cache[4] = NULL;
    return dout;
}

void description_attention(const attention_t *attn) {
    if (attn == NULL)
        return;

    printf("Attention(n_embd = %d, n_heads = %d)\n", attn->n_heads, attn->n_embd);
    printf("------------------------------------\n");
    printf("  n_embd : %d\n", attn->n_embd);
    printf("  n_heads: %d\n\n", attn->n_heads);
}   

int num_parameters_attention(const attention_t *attn) {
    return 0;
}   

void free_layer_attention(attention_t *attn) {
    if (attn == NULL)
        return;

    free_tensor(attn->buffer);
    free_tensor(attn->cache[0]);
    free_tensor(attn->cache[1]);
    free_tensor(attn->cache[2]);
    free_tensor(attn->cache[3]);
    free_tensor(attn->cache[4]);
    free(attn);
}

attention_t *Attention(int n_embd, int n_heads, int block_size) {

    if (n_embd % n_heads != 0){
        printf("Expected n_embd to be divisble by n_heads, but got %d % d == %f\n", n_embd, n_heads, n_embd % n_heads);
        return NULL;
    }

    attention_t *attn = (attention_t *)mallocCheck(sizeof(attention_t));

    // self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    int buffer_shape[2] = {block_size, block_size};
    attn->buffer = zeros(buffer_shape, 2);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j <= i; j++) {
            attn->buffer->t[i * block_size + j] = 1.0f;
        }
    }

    attn->cache[0] = NULL;
    attn->cache[1] = NULL;
    attn->cache[2] = NULL;
    attn->cache[3] = NULL;
    attn->cache[4] = NULL;
    attn->n_embd = n_embd;
    attn->n_heads = n_heads;
    attn->forward = forward_attention;
    attn->backward = backward_attention;
    attn->description = description_attention;
    attn->num_parameters = num_parameters_attention;
    attn->free_layer = free_layer_attention;
    return attn;
}