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

int main() {
    block_t *blk = Block(768, 12, 64, True);
    blk->description(blk);

    int x_shape[3] = {64, 64, 768};
    tensor_t *X = randn(x_shape, 3);
    tensor_t *out = blk->forward(blk, X);

    printf("X:\n");
    // print_tensor(X, 0);
    print_shape(X);

    printf("out:\n");
    // print_tensor(out, 0);
    print_shape(out);

    int global_grad_shape[3] = {64, 64, 768};
    tensor_t *global_grad = ones(global_grad_shape, 3);
    tensor_t *out_grad = blk->backward(blk, global_grad);

    printf("out.grad:\n");
    // print_tensor(out_grad, 0);
    print_shape(out_grad);

    // printf("self_attn.qkv.weight.grad:\n");
    // print_tensor(self_attn->qkv->dW, 0);
    // print_shape(self_attn->qkv->dW);

    // printf("self_attn.qkv.bias.grad:\n");
    // print_tensor(self_attn->qkv->db, 0);
    // print_shape(self_attn->qkv->db);

    // printf("self_attn.c_proj.weight.grad:\n");
    // print_tensor(self_attn->c_proj->dW, 0);
    // print_shape(self_attn->c_proj->dW);

    // printf("self_attn.c_proj.bias.grad:\n");
    // print_tensor(self_attn->c_proj->db, 0);
    // print_shape(self_attn->c_proj->db);

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