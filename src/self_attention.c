#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "blocks.h"

tensor_t *forward_self_attention(self_attention_t *self_attn, tensor_t *x) {

    if (self_attn == NULL) {
        printf("Expected required arugment *self_attn to be of type self_attention_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = x;
    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = self_attn->attn;
    qkv = self_attn->qkv;
    c_proj = self_attn->c_proj;

    out = qkv->forward(qkv, out);
    out = _attn->forward(_attn, out);
    out = c_proj->forward(c_proj, out);
    return out;
}

tensor_t *backward_self_attention(self_attention_t *self_attn, tensor_t *global_grad) {

    if (self_attn == NULL) {
        printf("Expected required arugment *self_attn to be of type self_attention_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = global_grad;
    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = self_attn->attn;
    qkv = self_attn->qkv;
    c_proj = self_attn->c_proj;

    out = c_proj->backward(c_proj, out);
    out = _attn->backward(_attn, out);
    out = qkv->backward(qkv, out);
    return out;
}

void description_self_attention(const self_attention_t *self_attn) {
    if (self_attn == NULL)
        return;

    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = self_attn->attn;
    qkv = self_attn->qkv;
    c_proj = self_attn->c_proj;

    printf("SelfAttention(n_embd = %d, n_heads = %d, block_size = %d, use_bias = %d)\n", self_attn->n_embd, self_attn->n_heads, self_attn->block_size, self_attn->use_bias);
    printf("------------------------------------------------------------------------\n\n");
    qkv->description(qkv);
    _attn->description(_attn);
    c_proj->description(c_proj);
}

int num_parameters_self_attention(const self_attention_t *self_attn) {
    if (self_attn == NULL)
        return 0;

    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = self_attn->attn;
    qkv = self_attn->qkv;
    c_proj = self_attn->c_proj;

    int total_parameters = 0;
    total_parameters += qkv->num_parameters(qkv);
    total_parameters += c_proj->num_parameters(c_proj);
    total_parameters += _attn->num_parameters(_attn);
    return total_parameters;
}

void free_layer_self_attention(self_attention_t *self_attn) {
    if (self_attn == NULL)
        return;

    attention_t *_attn;
    linear_t *qkv, *c_proj;
    _attn = self_attn->attn;
    qkv = self_attn->qkv;
    c_proj = self_attn->c_proj;

    _attn->free_layer(_attn);
    qkv->free_layer(qkv);
    c_proj->free_layer(c_proj);
    free(self_attn);
}

tensor_t **parameters_self_attention(const self_attention_t *self_attn) {
    if (self_attn == NULL)
        return NULL;

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * self_attn->_num_param_tensors);

    tensor_t **qkv_params = self_attn->qkv->parameters(self_attn->qkv);
    tensor_t **c_proj_params = self_attn->c_proj->parameters(self_attn->c_proj);

    int idx = 0;
    for (int i = 0; i < self_attn->qkv->_num_param_tensors; i++)
        parameters[idx++] = qkv_params[i];

    for (int i = 0; i < self_attn->c_proj->_num_param_tensors; i++)
        parameters[idx++] = c_proj_params[i];

    free(qkv_params);
    free(c_proj_params);

    return parameters;
}

tensor_t **gradients_self_attention(const self_attention_t *self_attn) {
    if (self_attn == NULL)
        return NULL;

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * self_attn->_num_param_tensors);

    tensor_t **qkv_grads = self_attn->qkv->gradients(self_attn->qkv);
    tensor_t **c_proj_grads = self_attn->c_proj->gradients(self_attn->c_proj);

    int idx = 0;
    for (int i = 0; i < self_attn->qkv->_num_param_tensors; i++)
        gradients[idx++] = qkv_grads[i];

    for (int i = 0; i < self_attn->c_proj->_num_param_tensors; i++)
        gradients[idx++] = c_proj_grads[i];

    free(qkv_grads);
    free(c_proj_grads);

    return gradients;
}

void load_state_dict_self_attention(self_attention_t *self_attn, tensor_t **state)
{
    if (self_attn == NULL)
    {
        printf("Expected required arugment *self_attn to be of type self_attention_t ptr, but got NULL.\n");
        return;
    }

    if (state == NULL)
    {
        printf("Expected required argument **state to be of type tensor_t ** ptr, but got NULL.\n");
        return;
    }

    self_attn->qkv->load_state_dict(self_attn->qkv, state);
    state += self_attn->qkv->_num_param_tensors;
    self_attn->c_proj->load_state_dict(self_attn->c_proj, state);
}

self_attention_t *SelfAttention(const int n_embd, const int n_heads, const int block_size, const int use_bias) {

    self_attention_t *self_attn = (self_attention_t*)mallocCheck(sizeof(self_attention_t));
    self_attn->n_embd = n_embd;
    self_attn->n_heads = n_heads;
    self_attn->use_bias = use_bias;
    self_attn->block_size = block_size;

    self_attn->qkv = Linear(n_embd, n_embd * 3, use_bias);
    self_attn->attn = Attention(n_embd, n_heads, block_size);
    self_attn->c_proj = Linear(n_embd, n_embd, use_bias);

    self_attn->forward = forward_self_attention;
    self_attn->backward = backward_self_attention;
    self_attn->description = description_self_attention;
    self_attn->num_parameters = num_parameters_self_attention;
    self_attn->free_layer = free_layer_self_attention;
    self_attn->parameters = parameters_self_attention;
    self_attn->gradients = gradients_self_attention;
    self_attn->load_state_dict = load_state_dict_self_attention;
    self_attn->_num_param_tensors = self_attn->qkv->_num_param_tensors + self_attn->c_proj->_num_param_tensors;
    return self_attn;
}