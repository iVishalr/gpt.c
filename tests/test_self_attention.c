#include <stdio.h>
#include "blocks.h"

int main()
{
    self_attention_t *self_attn = SelfAttention(6, 2, 2, True);
    self_attn->description(self_attn);

    int x_shape[3] = {4, 2, 6};
    tensor_t *X = randn(x_shape, 3);
    tensor_t *out = self_attn->forward(self_attn, X);

    printf("X:\n");
    print_tensor(X, 0);
    print_shape(X);

    printf("self_attn.qkv.weight:\n");
    print_tensor(self_attn->qkv->W, 0);
    print_shape(self_attn->qkv->W);

    printf("self_attn.qkv.bias:\n");
    print_tensor(self_attn->qkv->b, 0);
    print_shape(self_attn->qkv->b);

    printf("self_attn.c_proj.weight:\n");
    print_tensor(self_attn->c_proj->W, 0);
    print_shape(self_attn->c_proj->W);

    printf("self_attn.c_proj.bias:\n");
    print_tensor(self_attn->c_proj->b, 0);
    print_shape(self_attn->c_proj->b);

    printf("out:\n");
    print_tensor(out, 0);
    print_shape(out);

    int global_grad_shape[3] = {4, 2, 6};
    tensor_t *global_grad = ones(global_grad_shape, 3);
    tensor_t *out_grad = self_attn->backward(self_attn, global_grad);

    printf("out.grad:\n");
    print_tensor(out_grad, 0);
    print_shape(out_grad);

    printf("self_attn.qkv.weight.grad:\n");
    print_tensor(self_attn->qkv->dW, 0);
    print_shape(self_attn->qkv->dW);

    printf("self_attn.qkv.bias.grad:\n");
    print_tensor(self_attn->qkv->db, 0);
    print_shape(self_attn->qkv->db);

    printf("self_attn.c_proj.weight.grad:\n");
    print_tensor(self_attn->c_proj->dW, 0);
    print_shape(self_attn->c_proj->dW);

    printf("self_attn.c_proj.bias.grad:\n");
    print_tensor(self_attn->c_proj->db, 0);
    print_shape(self_attn->c_proj->db);

    self_attn->free_layer(self_attn);
    self_attn = NULL;
    free_tensor(out_grad);
    free_tensor(out);
    X = NULL;
    out = NULL;
    out_grad = NULL;
    return 0;
}