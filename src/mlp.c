#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "blocks.h"


tensor_t *forward_mlp(mlp_t *mlp, tensor_t *x);
tensor_t *backward_mlp(mlp_t *mlp, tensor_t *global_grad);
void description_mlp(const mlp_t *mlp);
int num_parameters_mlp(const mlp_t *mlp);
void free_layer_mlp(mlp_t *mlp);
void free_cache_mlp(mlp_t *mlp);
tensor_t **parameters_mlp(const mlp_t *mlp);
tensor_t **gradients_mlp(const mlp_t *mlp);
void load_state_dict_mlp(mlp_t *mlp, tensor_t **state);
void to_mlp(mlp_t *mlp, const device_t device);


// MLP Class
mlp_t *MLP(const int in_features, const int expansion_factor, const int use_bias) {

    mlp_t *mlp = (mlp_t *)mallocCheck(sizeof(mlp_t));
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
    mlp->free_cache = free_cache_mlp;
    mlp->parameters = parameters_mlp;
    mlp->gradients = gradients_mlp;
    mlp->load_state_dict = load_state_dict_mlp;
    mlp->to = to_mlp;
    mlp->_num_param_tensors = mlp->c_fc->_num_param_tensors + mlp->c_proj->_num_param_tensors;
    return mlp;
}


tensor_t *forward_mlp(mlp_t *mlp, tensor_t *x) {

    CHECK_ERROR(mlp == NULL, "Expected *mlp to be a mlp_t pointer, but got NULL.");
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");

    tensor_t *out = x;
    out = mlp->c_fc->forward(mlp->c_fc, out);
    out = mlp->gelu->forward(mlp->gelu, out);
    out = mlp->c_proj->forward(mlp->c_proj, out);
    return out;
}


tensor_t *backward_mlp(mlp_t *mlp, tensor_t *global_grad) {

    CHECK_ERROR(mlp == NULL, "Expected *mlp to be a mlp_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

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
    // printf("------------------------------------------------------------\n");
    // printf("Total Parameters: %d\n", mlp->num_parameters(mlp));
    // printf("------------------------------------------------------------\n");
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


void free_cache_mlp(mlp_t *mlp) {
    if (mlp == NULL)
        return;

    mlp->c_fc->free_cache(mlp->c_fc);
    mlp->gelu->free_cache(mlp->gelu);
    mlp->c_proj->free_cache(mlp->c_proj);
}


tensor_t **parameters_mlp(const mlp_t *mlp) {
    CHECK_ERROR(mlp == NULL, "Expected *mlp to be a mlp_t pointer, but got NULL.");

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * mlp->_num_param_tensors);
    
    tensor_t **c_fc_params = mlp->c_fc->parameters(mlp->c_fc);
    tensor_t **c_proj_params = mlp->c_proj->parameters(mlp->c_proj);

    int idx = 0;
    for (int i = 0; i < mlp->c_fc->_num_param_tensors; i++)
        parameters[idx++] = c_fc_params[i];
    
    for (int i = 0; i < mlp->c_proj->_num_param_tensors; i++)
        parameters[idx++] = c_proj_params[i];

    free(c_fc_params);
    free(c_proj_params);

    return parameters;
}


tensor_t **gradients_mlp(const mlp_t *mlp) {
    CHECK_ERROR(mlp == NULL, "Expected *mlp to be a mlp_t pointer, but got NULL.");

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * mlp->_num_param_tensors);

    tensor_t **c_fc_grads = mlp->c_fc->gradients(mlp->c_fc);
    tensor_t **c_proj_grads = mlp->c_proj->gradients(mlp->c_proj);

    int idx = 0;
    for (int i = 0; i < mlp->c_fc->_num_param_tensors; i++)
        gradients[idx++] = c_fc_grads[i];

    for (int i = 0; i < mlp->c_proj->_num_param_tensors; i++)
        gradients[idx++] = c_proj_grads[i];

    free(c_fc_grads);
    free(c_proj_grads);

    return gradients;
}


void load_state_dict_mlp(mlp_t *mlp, tensor_t **state) {
    CHECK_ERROR(mlp == NULL, "Expected *mlp to be a mlp_t pointer, but got NULL.");
    CHECK_ERROR(state == NULL, "Expected **state to be a tensor_t pointer, but got NULL.");

    mlp->c_fc->load_state_dict(mlp->c_fc, state);
    state += mlp->c_fc->_num_param_tensors;
    mlp->c_proj->load_state_dict(mlp->c_proj, state);
}


void to_mlp(mlp_t *mlp, const device_t device) {
    CHECK_ERROR(mlp == NULL, "Expected *mlp to be a mlp_t pointer, but got NULL.");

    linear_t *c_fc, *c_proj;
    gelu_t *gelu;

    c_fc = mlp->c_fc;
    c_proj = mlp->c_proj;
    gelu = mlp->gelu;

    c_fc->to(c_fc, device);
    c_proj->to(c_proj, device);
    gelu->to(gelu, device);
}
