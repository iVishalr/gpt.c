#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "dispatch.h"
#include "attention.h"


tensor_t *forward_attention(attention_t *attn, tensor_t *x);
tensor_t *backward_attention(attention_t *attn, tensor_t *global_grad);
void description_attention(const attention_t *attn);
int num_parameters_attention(const attention_t *attn);
void free_layer_attention(attention_t *attn);
void free_cache_attention(attention_t *attn);
void to_attention(attention_t *attn, const device_t device);


// Attention Class
attention_t *Attention(int n_embd, int n_heads, int block_size) {

    CHECK_ERROR(n_embd % n_heads != 0, "Expected n_embd to be divisible by n_heads, but got %d \% %d == %d\n", n_embd, n_heads, n_embd % n_heads);

    attention_t *attn = (attention_t *)mallocCheck(sizeof(attention_t));

    // self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    int buffer_shape[2] = {block_size, block_size};
    attn->buffer = zeros(buffer_shape, 2, CPU);

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
    attn->free_cache = free_cache_attention;
    attn->to = to_attention;
    return attn;
}


tensor_t *forward_attention(attention_t *attn, tensor_t *x) {

    CHECK_ERROR(attn == NULL, "Expected *attn to be a attention_t pointer, but got NULL.");
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");

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

    device_t device = x->device;
    int B, T, C, C3, n_heads, hs, buffer_row_size;
    B = x->shape[0];
    T = x->shape[1];
    C3 = x->shape[2];
    C = C3 / 3;
    n_heads = attn->n_heads;
    hs = C / n_heads;

    tensor_t *q, *k, *v; 
    tensor_t **cache = attn->cache;
    int qkv_transpose_shape[4] = {B, n_heads, T, hs};
    q = create_tensor(qkv_transpose_shape, 4, device);
    k = create_tensor(qkv_transpose_shape, 4, device);
    v = create_tensor(qkv_transpose_shape, 4, device);

    int att_shape[4] = {B, n_heads, T, T};
    tensor_t *att = create_tensor(att_shape, 4, device);
    tensor_t *preatt = create_tensor(att->shape, att->ndims, device);

    int out_shape[3] = {B, T, C};
    tensor_t *out = create_tensor(out_shape, 3, device);
    
    cache[0] = q;
    cache[1] = k;
    cache[2] = v;
    cache[3] = preatt;
    cache[4] = att;

    attention_forward_dispatch(x, attn->buffer, n_heads, cache, out);
    
    free_tensor(x);    
    return out;
}


tensor_t *backward_attention(attention_t *attn, tensor_t *global_grad) {

    CHECK_ERROR(attn == NULL, "Expected *attn to be a attention_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

    device_t device = global_grad->device;
    int B, T, C, n_heads, hs;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];
    n_heads = attn->n_heads;
    hs = C / n_heads;

    const tensor_t **cache = attn->cache;
    int dout_shape[3] = {B, T, C * 3};
    tensor_t *dout = create_tensor(dout_shape, 3, device);

    attention_backward_dispatch(global_grad, cache, n_heads, dout);

    free_tensor(global_grad);
    free_tensor(attn->cache[0]);
    free_tensor(attn->cache[1]);
    free_tensor(attn->cache[2]);
    free_tensor(attn->cache[3]);
    free_tensor(attn->cache[4]);
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


void free_cache_attention(attention_t *attn) {
    if (attn == NULL)
        return;

    free_tensor(attn->cache[0]);
    free_tensor(attn->cache[1]);
    free_tensor(attn->cache[2]);
    free_tensor(attn->cache[3]);
    free_tensor(attn->cache[4]);
    attn->cache[0] = NULL;
    attn->cache[1] = NULL;
    attn->cache[2] = NULL;
    attn->cache[3] = NULL;
    attn->cache[4] = NULL;
}


void to_attention(attention_t *attn, const device_t device) {
    CHECK_ERROR(attn == NULL, "Expected *attn to be a attention_t pointer, but got NULL.");

    attn->buffer->to(attn->buffer, device);
    attn->cache[0]->to(attn->cache[0], device);
    attn->cache[1]->to(attn->cache[1], device);
    attn->cache[2]->to(attn->cache[2], device);
    attn->cache[3]->to(attn->cache[3], device);
    attn->cache[4]->to(attn->cache[4], device);
}
