#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "dispatch.h"
#include "layer_norm.h"

#define DEFAULT_EPS 1e-5f


tensor_t *forward_layer_norm(layer_norm_t *norm, tensor_t *x);
tensor_t *backward_layer_norm(layer_norm_t *norm, tensor_t *global_grad);
void description_layer_norm(const layer_norm_t *norm);
int num_parameters_layer_norm(const layer_norm_t *norm);
void free_layer_layer_norm(layer_norm_t *norm);
void free_cache_layer_norm(layer_norm_t *norm);
tensor_t **parameters_layer_norm(const layer_norm_t *norm);
tensor_t **gradients_layer_norm(const layer_norm_t *norm);
void load_state_dict_layer_norm(layer_norm_t *norm, tensor_t **state);
void to_layer_norm(layer_norm_t *norm, const device_t device);


// LayerNorm Class
layer_norm_t *LayerNorm(int in_features, const float eps, const int use_bias) {

    CHECK_ERROR(in_features <= 0, "Expected in_features to be a positive integer, but got %d", in_features);

    layer_norm_t *norm = (layer_norm_t *)mallocCheck(sizeof(layer_norm_t));

    norm->cache[0] = NULL;
    norm->cache[1] = NULL;
    norm->cache[2] = NULL;
    norm->eps = eps != (float)DEFAULT_EPS ? eps : (float)DEFAULT_EPS;
    norm->use_bias = use_bias;
    norm->in_features = in_features;

    int param_shape[1] = {in_features};
    norm->W = ones(param_shape, 1, CPU);
    norm->b = use_bias > 0 ? zeros(param_shape, 1, CPU) : NULL;

    norm->dW = zeros(param_shape, 1, CPU);
    norm->db = use_bias > 0 ? zeros(param_shape, 1, CPU) : NULL;

    norm->forward = forward_layer_norm;
    norm->backward = backward_layer_norm;
    norm->description = description_layer_norm;
    norm->num_parameters = num_parameters_layer_norm;
    norm->free_layer = free_layer_layer_norm;
    norm->free_cache = free_cache_layer_norm;
    norm->parameters = parameters_layer_norm;
    norm->gradients = gradients_layer_norm;
    norm->load_state_dict = load_state_dict_layer_norm;
    norm->to = to_layer_norm;
    norm->_num_param_tensors = use_bias > 0 ? 2 : 1;
    return norm;
}


tensor_t *forward_layer_norm(layer_norm_t *norm, tensor_t *x) {
    
    CHECK_ERROR(norm == NULL, "Expected *norm to be a layer_norm_t pointer, but got NULL.");
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");

    device_t device = x->device;
    int B, T, in_features;
    B = x->shape[0];
    T = x->shape[1];
    in_features = x->shape[2];

    tensor_t *out = create_tensor(x->shape, x->ndims, device); // (B, T, C)
    norm->cache[0] = create_tensor(x->shape, x->ndims - 1, device); // (B, T)
    norm->cache[1] = create_tensor(x->shape, x->ndims - 1, device); // (B, T)

    layer_norm_forward_dispatch(norm->W, norm->b, x, norm->eps, norm->cache, out);

    norm->cache[2] = x;
    return out;
}


tensor_t *backward_layer_norm(layer_norm_t *norm, tensor_t *global_grad) {

    CHECK_ERROR(norm == NULL, "Expected *norm to be a layer_norm_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

    /*
        Backprop Analysis
        -----------------

                                       ((X - mean)
           layer_norm(X) =  W * ------------------------- + b
                                sqrt(variance ^ 2 + eps))

        Shape Analysis
        --------------
        X: (B, T, in_features)
        W: (in_features)
        b: (in_features)

        dW: (in_features)
        db: (in_features)
        dX: (B, T, in_features)
        global_grad: (B, T, in_features)

        rstd = sqrt(variance ^ 2 + eps)
        norm: (B,T,in_features) = (X - mean) / rstd

        db += 1 * global_grad.sum((0, 1))
        dW += (norm * global_grad).sum((0,1))
        
        dnorm = W * global_grad
        dval = dnorm - dnorm.mean(-1) - norm * (dnorm * dnorm).mean(-1)
        dval *= rstd
        dX += dval
    */

    device_t device = global_grad->device;
    int B, T, in_features;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    in_features = global_grad->shape[2];

    tensor_t *mean, *rstd, *x, *dout;
    mean = norm->cache[0];
    rstd = norm->cache[1];
    x = norm->cache[2];
    dout = zeros(x->shape, x->ndims, device);

    if (!norm->dW)
        norm->dW = zeros(norm->W->shape, norm->W->ndims, device);
    
    if (!norm->db)
        norm->db = norm->use_bias > 0 ? zeros(norm->b->shape, norm->b->ndims, device) : NULL;

    layer_norm_backward_dispatch(global_grad, norm->cache, norm->W, norm->dW, norm->db, dout);

    free_tensor(global_grad);
    free_tensor(norm->cache[0]);
    free_tensor(norm->cache[1]);
    free_tensor(norm->cache[2]);
    global_grad = NULL;
    norm->cache[0] = NULL;
    norm->cache[1] = NULL;
    norm->cache[2] = NULL;
    return dout;
}


void description_layer_norm(const layer_norm_t *norm) {
    
    int parameters = norm->W->length;
    
    if (norm->b)
        parameters += norm->b->length;

    char w_shape[1024], b_shape[1024];
    shape(norm->W, w_shape);
    shape(norm->b, b_shape);

    printf("LayerNorm\n");
    printf("---------\n");
    printf("in_features: %d\n", norm->in_features);
    printf("eps: %f\n", norm->eps);

    if (norm->use_bias > 0)
        printf("use_bias: True\n");
    else
        printf("use_bias: False\n");

    printf("total parameters: %d\n", parameters);
    printf("\tW [%s]: %d\n", w_shape, norm->W->length);
    if (norm->use_bias > 0)
        printf("\tb [%s]: %d\n", b_shape, norm->b->length);
    printf("\n");
}


int num_parameters_layer_norm(const layer_norm_t *norm) {
    if (norm == NULL)
        return 0;

    int total_parameters = norm->W->length;
    total_parameters += norm->use_bias > 0 ? norm->b->length : 0;
    return total_parameters;
}


void free_layer_layer_norm(layer_norm_t *norm) {
    if (norm == NULL)
        return;

    free_tensor(norm->dW);
    free_tensor(norm->db);
    free_tensor(norm->W);
    free_tensor(norm->b);
    free_tensor(norm->cache[0]);
    free_tensor(norm->cache[1]);
    free_tensor(norm->cache[2]);
    free(norm);    
}


void free_cache_layer_norm(layer_norm_t *norm) {
    if (norm == NULL)
        return;

    free_tensor(norm->cache[0]);
    free_tensor(norm->cache[1]);
    free_tensor(norm->cache[2]); 
    norm->cache[0] = NULL;
    norm->cache[1] = NULL;
    norm->cache[2] = NULL;
}

tensor_t **parameters_layer_norm(const layer_norm_t *norm) {
    CHECK_ERROR(norm == NULL, "Expected *norm to be a layer_norm_t pointer, but got NULL.");

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * norm->_num_param_tensors);
    parameters[0] = norm->W;
    if (norm->use_bias > 0) 
        parameters[1] = norm->b;
    return parameters;
}


tensor_t **gradients_layer_norm(const layer_norm_t *norm) {
    CHECK_ERROR(norm == NULL, "Expected *norm to be a layer_norm_t pointer, but got NULL.");

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * norm->_num_param_tensors);
    gradients[0] = norm->dW;
    if (norm->use_bias > 0)
        gradients[1] = norm->db;

    return gradients;
}


void load_state_dict_layer_norm(layer_norm_t *norm, tensor_t **state) {
    CHECK_ERROR(norm == NULL, "Expected *norm to be a layer_norm_t pointer, but got NULL.");
    CHECK_ERROR(state == NULL, "Expected **state to be a tensor_t pointer, but got NULL.");

    // check parameter and state length
    tensor_t *W = state[0];
    tensor_t *b = norm->use_bias > 0 ? state[0] : NULL;

    CHECK_ERROR(
        norm->W->length != W->length, 
        "Cannot load layer_norm weights. Expected a tensor of size %d, but got %d.", norm->W->length, W->length
    );
    CHECK_ERROR(
        norm->use_bias > 0 && norm->b->length != b->length, 
        "Cannot load layer_norm bias. Expected a tensor of size %d, but got %d.", norm->b->length, b->length
    );

    memcpy(norm->W->t, W->t, norm->W->length * sizeof(float));
    if (norm->use_bias > 0)
        memcpy(norm->b->t, b->t, norm->b->length * sizeof(float));
}


void to_layer_norm(layer_norm_t *norm, const device_t device) {
    CHECK_ERROR(norm == NULL, "Expected *norm to be a layer_norm_t pointer, but got NULL.");

    norm->W->to(norm->W, device);
    norm->b->to(norm->b, device);
    norm->dW->to(norm->dW, device);
    norm->db->to(norm->db, device);
    norm->cache[0]->to(norm->cache[0], device);
    norm->cache[1]->to(norm->cache[1], device);
    norm->cache[2]->to(norm->cache[2], device);
}
