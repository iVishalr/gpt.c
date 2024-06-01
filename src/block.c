#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include "utils.h"
#include "blocks.h"


tensor_t *forward_block(block_t *blk, tensor_t *x);
tensor_t *backward_block(block_t *blk, tensor_t *global_grad); 
void description_block(const block_t *blk);
int num_parameters_block(const block_t *blk); 
void free_layer_block(block_t *blk);
tensor_t **parameters_block(const block_t *blk);
tensor_t **gradients_block(const block_t *blk);
void load_state_dict_block(block_t *blk, tensor_t **state);


// Block Class
block_t *Block(const int n_embd, const int n_heads, const int block_size, const int use_bias) {
    block_t *blk = (block_t *)mallocCheck(sizeof(block_t));

    blk->n_embd = n_embd;
    blk->n_heads = n_heads;
    blk->block_size = block_size;
    blk->use_bias = use_bias;

    blk->ln1 = LayerNorm(n_embd, 1e-5, use_bias);
    blk->attn = SelfAttention(n_embd, n_heads, block_size, use_bias);
    blk->ln2 = LayerNorm(n_embd, 1e-5, use_bias);
    blk->mlp = MLP(n_embd, 4, use_bias);

    blk->forward = forward_block;
    blk->backward = backward_block;
    blk->description = description_block;
    blk->free_layer = free_layer_block;
    blk->num_parameters = num_parameters_block;

    blk->parameters = parameters_block;
    blk->gradients = gradients_block;
    blk->load_state_dict = load_state_dict_block;

    blk->_num_param_tensors = 0;
    blk->_num_param_tensors += blk->ln1->_num_param_tensors;
    blk->_num_param_tensors += blk->attn->_num_param_tensors;
    blk->_num_param_tensors += blk->ln2->_num_param_tensors;
    blk->_num_param_tensors += blk->mlp->_num_param_tensors;
    return blk;
}


tensor_t *forward_block(block_t *blk, tensor_t *x) {

    if (blk == NULL) {
        printf("Expected required arugment *blk to be of type block_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    tensor_t *resid = create_tensor(x->shape, x->ndims);
    tensor_copy(resid, x);

    tensor_t *out = x;
    out = ln1->forward(ln1, out);
    out = attn->forward(attn, out);

    // out = resid + out
    cblas_saxpy(out->length, 1.0f, resid->t, 1, out->t, 1);

    tensor_copy(resid, out);

    out = ln2->forward(ln2, out);
    out = mlp->forward(mlp, out);

    // out = resid + out
    cblas_saxpy(out->length, 1.0f, resid->t, 1, out->t, 1);
    free_tensor(resid);

    return out;
}


tensor_t *backward_block(block_t *blk, tensor_t *global_grad) {
    
    if (blk == NULL) {
        printf("Expected required arugment *blk to be of type block_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    tensor_t *out;
    tensor_t *gg1 = create_tensor(global_grad->shape, global_grad->ndims);
    tensor_t *gg2 = create_tensor(global_grad->shape, global_grad->ndims);
    tensor_copy(gg1, global_grad);

    out = mlp->backward(mlp, gg1);
    out = ln2->backward(ln2, out);

    // out = global grad + out
    cblas_saxpy(out->length, 1.0f, global_grad->t, 1, out->t, 1);
    free_tensor(global_grad);
    
    tensor_copy(gg2, out);

    out = attn->backward(attn, out);
    out = ln1->backward(ln1, out);

    // out = gg2 + out
    cblas_saxpy(out->length, 1.0f, gg2->t, 1, out->t, 1);
    free_tensor(gg2);

    return out;
}


void description_block(const block_t *blk) {
    if (blk == NULL)
        return;

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    printf("Block(n_embd = %d, n_heads = %d, block_size = %d, use_bias = %d)\n", blk->n_embd, blk->n_heads, blk->block_size, blk->use_bias);
    printf("----------------------------------------------------------------\n\n");
    ln1->description(ln1);
    attn->description(attn);
    ln2->description(ln2);
    mlp->description(mlp);
}


int num_parameters_block(const block_t *blk) {
    if (blk == NULL) 
        return 0;

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    int total_parameters = 0;
    total_parameters += ln1->num_parameters(ln1);
    total_parameters += ln1->num_parameters(ln2);
    total_parameters += attn->num_parameters(attn);
    total_parameters += mlp->num_parameters(mlp);
    return total_parameters;
}


void free_layer_block(block_t *blk) {
    if (blk == NULL)
        return;

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    ln1->free_layer(ln1);
    ln2->free_layer(ln2);
    attn->free_layer(attn);
    mlp->free_layer(mlp);
    free(blk);
}


tensor_t **parameters_block(const block_t *blk) {
    if (blk == NULL)
        return NULL;

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * blk->_num_param_tensors);
    tensor_t **ln1_params = ln1->parameters(ln1);
    tensor_t **attn_params = attn->parameters(attn);
    tensor_t **ln2_params = ln2->parameters(ln2);
    tensor_t **mlp_params = mlp->parameters(mlp);

    int idx = 0;
    for (int i = 0; i < ln1->_num_param_tensors; i++)
        parameters[idx++] = ln1_params[i];

    for (int i = 0; i < attn->_num_param_tensors; i++)
        parameters[idx++] = attn_params[i];

    for (int i = 0; i < ln2->_num_param_tensors; i++)
        parameters[idx++] = ln2_params[i];

    for (int i = 0; i < mlp->_num_param_tensors; i++)
        parameters[idx++] = mlp_params[i];

    free(ln1_params);
    free(attn_params);
    free(ln2_params);
    free(mlp_params);
    return parameters;
}


tensor_t **gradients_block(const block_t *blk) {
    if (blk == NULL)
        return NULL;

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * blk->_num_param_tensors);
    tensor_t **ln1_grads = ln1->gradients(ln1);
    tensor_t **attn_grads = attn->gradients(attn);
    tensor_t **ln2_grads = ln2->gradients(ln2);
    tensor_t **mlp_grads = mlp->gradients(mlp);

    int idx = 0;
    for (int i = 0; i < ln1->_num_param_tensors; i++)
        gradients[idx++] = ln1_grads[i];

    for (int i = 0; i < attn->_num_param_tensors; i++)
        gradients[idx++] = attn_grads[i];

    for (int i = 0; i < ln2->_num_param_tensors; i++)
        gradients[idx++] = ln2_grads[i];

    for (int i = 0; i < mlp->_num_param_tensors; i++)
        gradients[idx++] = mlp_grads[i];

    free(ln1_grads);
    free(attn_grads);
    free(ln2_grads);
    free(mlp_grads);
    return gradients;
}


void load_state_dict_block(block_t *blk, tensor_t **state) {
    if (blk == NULL)
    {
        printf("Expected required arugment *blk to be of type block_t ptr, but got NULL.\n");
        return;
    }

    if (state == NULL)
    {
        printf("Expected required argument **state to be of type tensor_t ** ptr, but got NULL.\n");
        return;
    }

    self_attention_t *attn;
    layer_norm_t *ln1, *ln2;
    mlp_t *mlp;

    ln1 = blk->ln1;
    ln2 = blk->ln2;
    attn = blk->attn;
    mlp = blk->mlp;

    ln1->load_state_dict(ln1, state);
    state += ln1->_num_param_tensors;
    attn->load_state_dict(attn, state);
    state += attn->_num_param_tensors;
    ln2->load_state_dict(ln2, state);
    state += ln2->_num_param_tensors;
    mlp->load_state_dict(mlp, state);
}
