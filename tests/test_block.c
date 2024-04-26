#include <stdio.h>
#include "linear.h"
#include "activation.h"
#include "loss.h"
#include "layer_norm.h"
#include "embedding.h"
#include "attention.h"
#include "blocks.h"

#define True 1
#define False 0

int main()
{
    block_t *blk = Block(6, 2, 2, True);
    blk->description(blk);

    int x_shape[3] = {4, 2, 6};
    tensor_t *X = randn(x_shape, 3);

    printf("X:\n");
    print_tensor(X, 0);
    print_shape(X);

    printf("blk.attn.qkv.weight:\n");
    print_tensor(blk->attn->qkv->W, 0);
    print_shape(blk->attn->qkv->W);

    printf("blk.attn.qkv.bias:\n");
    print_tensor(blk->attn->qkv->b, 0);
    print_shape(blk->attn->qkv->b);

    printf("blk.attn.c_proj.weight:\n");
    print_tensor(blk->attn->c_proj->W, 0);
    print_shape(blk->attn->c_proj->W);

    printf("blk.attn.c_proj.bias:\n");
    print_tensor(blk->attn->c_proj->b, 0);
    print_shape(blk->attn->c_proj->b);

    printf("blk.ln1.weight:\n");
    print_tensor(blk->ln1->W, 0);
    print_shape(blk->ln1->W);

    printf("blk.ln1.bias:\n");
    print_tensor(blk->ln1->b, 0);
    print_shape(blk->ln1->b);

    printf("blk.ln2.weight:\n");
    print_tensor(blk->ln2->W, 0);
    print_shape(blk->ln2->W);

    printf("blk.ln2.bias:\n");
    print_tensor(blk->ln2->b, 0);
    print_shape(blk->ln2->b);

    printf("blk.mlp.c_fc.weight:\n");
    print_tensor(blk->mlp->c_fc->W, 0);
    print_shape(blk->mlp->c_fc->W);

    printf("blk.mlp.c_fc.bias:\n");
    print_tensor(blk->mlp->c_fc->b, 0);
    print_shape(blk->mlp->c_fc->b);

    printf("blk.mlp.c_proj.weight:\n");
    print_tensor(blk->mlp->c_proj->W, 0);
    print_shape(blk->mlp->c_proj->W);

    printf("blk.mlp.c_proj.bias:\n");
    print_tensor(blk->mlp->c_proj->b, 0);
    print_shape(blk->mlp->c_proj->b);

    tensor_t *out = blk->forward(blk, X);

    printf("out:\n");
    print_tensor(out, 0);
    print_shape(out);

    int global_grad_shape[3] = {4, 2, 6};
    tensor_t *global_grad = ones(global_grad_shape, 3);
    tensor_t *out_grad = blk->backward(blk, global_grad);

    printf("out.grad:\n");
    print_tensor(out_grad, 0);
    print_shape(out_grad);

    printf("blk.attn.qkv.weight.grad:\n");
    print_tensor(blk->attn->qkv->dW, 0);
    print_shape(blk->attn->qkv->dW);

    printf("blk.attn.qkv.bias.grad:\n");
    print_tensor(blk->attn->qkv->db, 0);
    print_shape(blk->attn->qkv->db);

    printf("blk.attn.c_proj.weight.grad:\n");
    print_tensor(blk->attn->c_proj->dW, 0);
    print_shape(blk->attn->c_proj->dW);

    printf("blk.attn.c_proj.bias.grad:\n");
    print_tensor(blk->attn->c_proj->db, 0);
    print_shape(blk->attn->c_proj->db);

    printf("blk.ln1.weight.grad:\n");
    print_tensor(blk->ln1->dW, 0);
    print_shape(blk->ln1->dW);

    printf("blk.ln1.bias.grad:\n");
    print_tensor(blk->ln1->db, 0);
    print_shape(blk->ln1->db);

    printf("blk.ln2.weight.grad:\n");
    print_tensor(blk->ln2->dW, 0);
    print_shape(blk->ln2->dW);

    printf("blk.ln1.bias.grad:\n");
    print_tensor(blk->ln2->db, 0);
    print_shape(blk->ln2->db);

    printf("blk.mlp.c_fc.weight.grad:\n");
    print_tensor(blk->mlp->c_fc->dW, 0);
    print_shape(blk->mlp->c_fc->dW);

    printf("blk.mlp.c_fc.bias.grad:\n");
    print_tensor(blk->mlp->c_fc->db, 0);
    print_shape(blk->mlp->c_fc->db);

    printf("blk.mlp.c_proj.weight.grad:\n");
    print_tensor(blk->mlp->c_proj->dW, 0);
    print_shape(blk->mlp->c_proj->dW);

    printf("blk.mlp.c_proj.bias.grad:\n");
    print_tensor(blk->mlp->c_proj->db, 0);
    print_shape(blk->mlp->c_proj->db);

    blk->free_layer(blk);
    blk = NULL;
    // free_tensor(X);
    free_tensor(out_grad);
    free_tensor(out);
    X = NULL;
    out = NULL;
    out_grad = NULL;
    return 0;
}