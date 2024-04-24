#include <stdio.h>
#include <stdlib.h>
#include "blocks.h"

tensor_t *forward_self_attention(self_attention_t *attn, tensor_t *x) {

    if (attn == NULL) {
        printf("Expected required arugment *attn to be of type self_attention_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = x;
    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = attn->attn;
    qkv = attn->qkv;
    c_proj = attn->c_proj;

    out = qkv->forward(qkv, out);
    out = _attn->forward(_attn, out);
    out = c_proj->forward(c_proj, out);
    return out;
}

tensor_t *backward_self_attention(self_attention_t *attn, tensor_t *global_grad) {

    if (attn == NULL) {
        printf("Expected required arugment *attn to be of type self_attention_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = global_grad;
    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = attn->attn;
    qkv = attn->qkv;
    c_proj = attn->c_proj;

    out = c_proj->backward(c_proj, out);
    out = _attn->backward(_attn, out);
    out = qkv->backward(qkv, out);
    return out;
}

void description_self_attention(const self_attention_t *attn) {
    if (attn == NULL)
        return;

    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = attn->attn;
    qkv = attn->qkv;
    c_proj = attn->c_proj;

    printf("SelfAttention(n_embd = %d, n_heads = %d, block_size = %d, use_bias = %d)\n", attn->n_embd, attn->n_heads, attn->block_size, attn->use_bias);
    printf("------------------------------------------------------------------------\n\n");
    qkv->description(qkv);
    _attn->description(_attn);
    c_proj->description(c_proj);
    printf("------------------------------------------------------------------------\n");
    printf("Parameters: %d\n", attn->num_parameters(attn));
    printf("------------------------------------------------------------------------\n");
}

int num_parameters_self_attention(const self_attention_t *attn) {
    if (attn == NULL)
        return 0;

    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = attn->attn;
    qkv = attn->qkv;
    c_proj = attn->c_proj;

    int total_parameters = 0;
    total_parameters += qkv->num_parameters(qkv);
    total_parameters += c_proj->num_parameters(c_proj);
    total_parameters += _attn->num_parameters(_attn);
    return total_parameters;
}

void free_layer_self_attention(self_attention_t *attn) {
    if (attn == NULL)
        return;

    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = attn->attn;
    qkv = attn->qkv;
    c_proj = attn->c_proj;

    _attn->free_layer(_attn);
    qkv->free_layer(qkv);
    c_proj->free_layer(c_proj);
    free(attn);
}

self_attention_t *SelfAttention(const int n_embd, const int n_heads, const int block_size, const int use_bias) {

    self_attention_t *attn = (self_attention_t*)malloc(sizeof(self_attention_t));
    attn->n_embd = n_embd;
    attn->n_heads = n_heads;
    attn->use_bias = use_bias;
    attn->block_size = block_size;

    attn->qkv = Linear(n_embd, n_embd * 3, use_bias);
    attn->attn = Attention(n_embd, n_heads, block_size);
    attn->c_proj = Linear(n_embd, n_embd, use_bias);

    attn->forward = forward_self_attention;
    attn->backward = backward_self_attention;
    attn->description = description_self_attention;
    attn->num_parameters = num_parameters_self_attention;
    attn->free_layer = free_layer_self_attention;
    return attn;
}