#include <stdio.h>
#include "transformer.h"
#include "optim.h"
#include "dataloader.h"

#define True 1
#define False 0

int main() {

    GPT2Config_t gpt2_config;
    gpt2_config.block_size = 1024;
    gpt2_config.vocab_size = 50257;
    gpt2_config.n_embd = 768;
    gpt2_config.n_heads = 12;
    gpt2_config.n_layers = 1;

    gpt2_t *gpt = GPT2(&gpt2_config);


    gpt->description(gpt);
    printf("Number of Parameter Tensors: %d\n", gpt->_num_param_tensors);

    int x_shape[2] = {4, 2};
    tensor_t *X = ones(x_shape, 2);

    printf("X:\n");
    print_shape(X);

    tensor_t *out = gpt->forward(gpt, X, NULL);

    printf("out:\n");
    print_shape(out);

    tensor_t *global_grad = ones(out->shape, out->ndims);
    tensor_t *out_grad = gpt->backward(gpt, global_grad);

    printf("out.grad:\n");
    print_shape(out_grad);

    adamW_t *optimizer = AdamW(gpt->parameters(gpt), gpt->gradients(gpt), gpt->_num_param_tensors, 1e-4f, 0.9f, 0.99f, 1e-08f, 0.01);
    gpt->free_layer(gpt);
    optimizer->free_layer(optimizer);

    dataloader_t *loader = DataLoader("data/tiny_shakespeare.txt", 64, 1024);
    loader->free_layer(loader);

    free_tensor(out_grad);
    free_tensor(out);
    X = NULL;
    out = NULL;
    out_grad = NULL;
    return 0;
}