#include <stdio.h>
#include "transformer.h"

#define True 1
#define False 0

int main() {

    GPT2Config_t gpt2_config;
    gpt2_config.block_size = 2;
    gpt2_config.vocab_size = 5;
    gpt2_config.n_embd = 6;
    gpt2_config.n_heads = 2;
    gpt2_config.n_layers = 1;

    gpt2_t *gpt = GPT2(&gpt2_config);
    gpt->description(gpt);

    int x_shape[2] = {4, 2};
    tensor_t *X = ones(x_shape, 2);

    printf("X:\n");
    print_tensor(X, 0);
    print_shape(X);

    tensor_t *out = gpt->forward(gpt, X, NULL);

    printf("out:\n");
    print_tensor(out, 0);
    print_shape(out);

    tensor_t *global_grad = ones(out->shape, out->ndims);
    tensor_t *out_grad = gpt->backward(gpt, global_grad);

    printf("out.grad:\n");
    print_tensor(out_grad, 0);
    print_shape(out_grad);

    gpt->free_layer(gpt);
    free_tensor(out_grad);
    free_tensor(out);
    X = NULL;
    out = NULL;
    out_grad = NULL;
    return 0;
}