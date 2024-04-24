#include <stdio.h>
#include <stdlib.h>
#include "blocks.h"

tensor_t *forward_mlp(mlp_t *mlp, tensor_t *x) {

    if (mlp == NULL) {
        printf("Expected required arugment *mlp to be of type mlp_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = x;
    out = mlp->c_fc->forward(mlp->c_fc, out);
    out = mlp->gelu->forward(mlp->gelu, out);
    out = mlp->c_proj->forward(mlp->c_proj, out);
    return out;
}

tensor_t *backward_mlp(mlp_t *mlp, tensor_t *global_grad) {

    if (mlp == NULL) {
        printf("Expected required arugment *mlp to be of type mlp_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = global_grad;
    out = mlp->c_proj->backward(mlp->c_proj, out);
    out = mlp->gelu->backward(mlp->gelu, out);
    out = mlp->c_fc->backward(mlp->c_fc, out);
    return out;
}

void description_mlp(const mlp_t *mlp) {
    if (mlp == NULL)
        return;

    linear_t *c_fc, *c_proj;
    gelu_t *gelu;

    c_fc = mlp->c_fc;
    c_proj = mlp->c_proj;
    gelu = mlp->gelu;

    printf("MLP(in_features = %d, expansion_factor = %d, use_bias = %d)\n", mlp->in_features, mlp->expansion_factor, mlp->use_bias);
    printf("-----------------------------------------------------------\n");
    c_fc->description(c_fc);
    gelu->description(gelu);
    c_proj->description(c_proj);
    printf("------------------------------------------------------------\n");
    printf("Total Parameters: %d\n", mlp->num_parameters(mlp));
    printf("------------------------------------------------------------\n");
}

int num_parameters_mlp(const mlp_t *mlp) {
    if (mlp == NULL)
        return 0;

    int total_parameters = 0;
    total_parameters += mlp->c_fc->num_parameters(mlp->c_fc);
    total_parameters += mlp->c_proj->num_parameters(mlp->c_proj);
    return total_parameters;
}

void free_layer_mlp(mlp_t *mlp) {
    if (mlp == NULL)
        return;

    mlp->c_fc->free_layer(mlp->c_fc);
    mlp->gelu->free_layer(mlp->gelu);
    mlp->c_proj->free_layer(mlp->c_proj);
    mlp->c_fc = NULL;
    mlp->c_proj = NULL;
    free(mlp);
}

mlp_t *MLP(const int in_features, const int expansion_factor, const int use_bias) {

    mlp_t *mlp = (mlp_t*)malloc(sizeof(mlp_t));
    mlp->in_features = in_features;
    mlp->expansion_factor = expansion_factor;
    mlp->use_bias = use_bias;

    mlp->c_fc = Linear(in_features, expansion_factor * in_features, use_bias);
    mlp->gelu = GELU();
    mlp->c_proj = Linear(expansion_factor * in_features, in_features, use_bias);

    mlp->forward = forward_mlp;
    mlp->backward = backward_mlp;
    mlp->description = description_mlp;
    mlp->num_parameters = num_parameters_mlp;
    mlp->free_layer = free_layer_mlp;
    return mlp;
}