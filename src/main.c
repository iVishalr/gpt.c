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
    self_attention_t *self_attn = SelfAttention(768, 12, 64, True);
    self_attn->description(self_attn);

    int x_shape[3] = {64, 64, 768 * 3};
    tensor_t *X = randn(x_shape, 3);
    tensor_t *out = self_attn->forward(self_attn, X);

    printf("X:\n");
    print_shape(X);

    printf("out:\n");
    print_shape(out);

    int global_grad_shape[3] = {64, 64, 768};
    tensor_t *global_grad = ones(global_grad_shape, 3);
    tensor_t *out_grad = self_attn->backward(self_attn, global_grad);

    printf("out.grad:\n");
    print_shape(out_grad);

    self_attn->free_layer(self_attn);
    self_attn = NULL;
    free_tensor(X);
    free_tensor(out_grad);
    free_tensor(out);
    X = NULL;
    out = NULL;
    out_grad = NULL;
    return 0;
}

int main1() {
    // embedding_t *embedding = Embedding(64, 128);
    // layer_norm_t *norm = LayerNorm(128, 1e-5f, True);
    // linear_t *layer = Linear(128, 256, True);
    // gelu_t *gelu = GELU();
    // softmax_t *softmax = Softmax();
    // cross_entropy_loss_t *loss = CrossEntropyLoss();

    // embedding->description(embedding);
    // norm->description(norm);
    // layer->description(layer);
    // gelu->description(gelu);
    // softmax->description(softmax);
    // loss->description(loss);

    // int x_shape[2] = {4, 64};
    // int targets_shape[2] = {4, 256};
    // tensor_t *x = ones(x_shape, 2);
    // tensor_t *targets = ones(targets_shape, 2);

    // printf("X:\n");
    // print_tensor(x);
    // printf("Targets:\n");
    // print_tensor(targets);

    // printf("embedding.weight:\n");
    // print_tensor(embedding->W);

    // printf("linear.weight:\n");
    // print_tensor(layer->W);

    // printf("linear.bias:\n");
    // print_tensor(layer->b);

    // printf("FORWARD\n");
    // tensor_t *out, *ret;
    // out = embedding->forward(embedding, x);
    // ret = out;
    // out = norm->forward(norm, out);
    // free_tensor(ret);
    // ret = out;
    // out = layer->forward(layer, out);
    // free_tensor(ret);
    // ret = out;
    // out = gelu->forward(gelu, out);
    // free_tensor(ret);
    // ret = out;
    // out = softmax->forward(softmax, out);
    // free_tensor(ret);
    // ret = out;
    // tensor_t *out_loss = loss->forward(loss, out, targets);


    // printf("logits:\n");
    // print_tensor(out);
    // print_shape(out);

    // printf("loss:\n");
    // print_tensor(out_loss);
    // print_shape(out_loss);

    // printf("BACKPROP\n");
    // int grad_shape[2] = {4, 256};
    // tensor_t *global_grad = ones(grad_shape, 2);
    // tensor_t *out_grad = loss->backward(loss, global_grad);
    // out_grad = gelu->backward(gelu, out_grad);
    // out_grad = layer->backward(layer, out_grad);
    // out_grad = norm->backward(norm, out_grad);
    // out_grad = embedding->backward(embedding, out_grad);

    // printf("dout:\n");
    // print_tensor(out_grad);

    // printf("linear.weight.grad:\n");
    // print_tensor(layer->dW);

    // printf("linear.bias.grad:\n");
    // print_tensor(layer->db);

    // printf("\nnorm.weight.grad:\n");
    // print_tensor(norm->dW);

    // printf("\nnorm.bias.grad:\n");
    // print_tensor(norm->db);

    // printf("\nembedding.weight.grad:\n");
    // print_tensor(embedding->dW);

    // printf("Layer Freed!\n");
    // layer->free_layer(layer);
    // printf("GELU Freed!\n");
    // gelu->free_layer(gelu);
    // printf("Softmax Freed!\n");
    // softmax->free_layer(softmax);
    // printf("CrossEntropy Freed!\n");
    // loss->free_layer(loss);
    // printf("LayerNorm Freed!\n");
    // norm->free_layer(norm);
    // printf("Embedding Freed!\n");
    // embedding->free_layer(embedding);

    // free_tensor(x);
    // free_tensor(out);
    // free_tensor(out_loss);
    // free_tensor(out_grad);
    // free_tensor(targets);
    return 0;
}
