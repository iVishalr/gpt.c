#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include "blocks.h"

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
    // printf("----------------------------------------------------------------\n");
    // printf("Parameters: %d\n", blk->num_parameters(blk));
    // printf("----------------------------------------------------------------\n");
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

block_t *Block(const int n_embd, const int n_heads, const int block_size, const int use_bias) {
    block_t *blk = (block_t*)malloc(sizeof(block_t));

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
    return blk;
}