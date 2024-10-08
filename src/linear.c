#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>
#include "utils.h"
#include "linear.h"


void _calculate_fan_in_and_fan_out(tensor_t *t, int *result);
void kaiming_uniform(tensor_t *t, float a, const char *mode, const char *non_linearity);
tensor_t *forward_linear(linear_t *linear, tensor_t *x);
tensor_t *backward_linear(linear_t *linear, tensor_t *global_grad);
void description_linear(const linear_t *linear);
int num_parameters_linear(const linear_t *linear);
void free_layer_linear(linear_t *linear);
void free_cache_linear(linear_t *linear);
tensor_t **parameters_linear(const linear_t *linear);
tensor_t **gradients_linear(const linear_t *linear);
void load_state_dict_linear(linear_t *linear, tensor_t **state);


// Linear Class
linear_t *Linear(const int in_features, const int out_features, const int use_bias)
{
    linear_t *linear = (linear_t *)mallocCheck(sizeof(linear_t));
    int w_shape[2] = {out_features, in_features};
    int b_shape[1] = {out_features};

    tensor_t *W = zeros(w_shape, 2);
    tensor_t *b = use_bias > 0 ? zeros(b_shape, 1) : NULL;

    // init weights
    kaiming_uniform(W, sqrtf(5.0f), "fan_in", "leaky_relu");

    if (b != NULL)
    {
        int result[2];
        _calculate_fan_in_and_fan_out(W, result);
        int fan_in = result[0];
        float bound = fan_in > 0.0f ? 1.0f / sqrtf(fan_in) : 0.0f;
        uniform(b, -bound, bound);
    }

    linear->W = W;
    linear->b = b;
    linear->in_features = in_features;
    linear->out_features = out_features;
    linear->use_bias = use_bias;
    linear->dW = zeros(w_shape, 2);
    linear->db = use_bias > 0 ? zeros(b_shape, 1) : NULL;
    linear->cache = NULL;
    linear->forward = forward_linear;
    linear->backward = backward_linear;
    linear->description = description_linear;
    linear->free_layer = free_layer_linear;
    linear->free_cache = free_cache_linear;
    linear->num_parameters = num_parameters_linear;
    linear->parameters = parameters_linear;
    linear->gradients = gradients_linear;
    linear->load_state_dict = load_state_dict_linear;
    linear->_num_param_tensors = use_bias > 0 ? 2 : 1;
    return linear;
}


void _calculate_fan_in_and_fan_out(tensor_t *t, int *result)
{

    if (t == NULL) {
        printf("Expected required arugment *t to be of type tensor_t ptr, but got NULL.\n");
        return;
    }

    if (result == NULL) {
        printf("Expected required arugment *result to be of type int ptr, but got NULL.\n");
        return;
    }

    int ndims = t->ndims;
    if (ndims < 2) {
        printf("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions.\n");
        return;
    }

    int num_input_fmaps = t->shape[1];
    int num_output_fmaps = t->shape[0];
    int receptive_field_size = 1;
    
    if (ndims > 2) {
        for (int i = 2; i < ndims; i++)
            receptive_field_size *= t->shape[i];
    }
    
    int fan_in = num_input_fmaps * receptive_field_size;
    int fan_out = num_output_fmaps * receptive_field_size;
    result[0] = fan_in;
    result[1] = fan_out;
}


void kaiming_uniform(tensor_t *t, float a, const char *mode, const char *non_linearity) {
    int result[2];
    _calculate_fan_in_and_fan_out(t, result);

    int fan;
    if (strcmp(mode, "fan_in") == 0) {
        fan = result[0];
    } else {
        fan = result [1];
    }

    float negative_slope = a;
    if (a == 0.0f)
        negative_slope = 0.01f;
    
    float gain = sqrtf(2.0f / (1.0f + powf(negative_slope, 2.0f)));
    float std = gain / sqrtf(fan);
    float bound = sqrtf(3.0f) * std;
    uniform(t, -bound, bound);
}


tensor_t *forward_linear(linear_t *linear, tensor_t *x) {
    
    if (linear == NULL) {
        printf("Expected required arugment *linear to be of type linear_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    // Shape Analysis
    // --------------
    // x: (B, T, in_features)
    // W: (out_features, in_features)
    // b: (out_features)
    // y = x @ W.T + b
    // y: (B * T, in_features) @ (in_features, out_features) -> (B * T, out_features) + (out_features)
        
    int B, T, in_features;
    B = x->shape[0];
    T = x->shape[1];
    in_features = x->shape[2];

    tensor_t *out, *ret;

    int out_shape[3] = {B, T, linear->out_features};
    out = create_tensor(out_shape, 3);

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        B * T, linear->out_features, linear->in_features, 
        1.0f, x->t, linear->in_features, 
        linear->W->t, linear->in_features, 
        0.0f, out->t, linear->out_features
    );

    // add bias to out tensor (B * T, out_features) + (out_features)
    if (linear->b) {
        int row_size = linear->out_features;
        for (int row = 0; row < B * T; row++) {
            for (int j = 0; j < row_size; j++) {
                out->t[row * row_size + j] += linear->b->t[j];
            }
        }
    }

    // cache the input to the layer
    linear->cache = x;
    return out;
}


tensor_t *backward_linear(linear_t *linear, tensor_t *global_grad) {
    
    if (linear == NULL) {
        printf("Expected required arugment *linear to be of type linear_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    /*
        Backprop Analysis
        -----------------

        In forward pass, we perform the function: y = x @ W.R + b
        For the above function, we have three inputs namely, x, W, b.
        During backpropagation, we calculate the derivatives of function w.r.t inputs.
        In backward pass for Linear layer, we need to calculate the gradients to
        x (input to the layer), W (weights of the layer) and b (bias of the layer).
        Let's call these dx, dW and db respectively.

        y = x @ W.T + b
        dx = W.T
        dW = x
        db = 1

        dx, dW, db are called as local gradients. In order to form the chain rule,
        we need to multiply the local gradients with the global gradients.

        dx += grad @ W
        dW += grad @ x
        db += grad

        Shape Analysis
        --------------
        If input to Linear was of the shape, (B, T, in_features) and W (out_features, in_features) and b (out_features) then,
        dx = (B, T, in_features)
        dW = (out_features, in_features)
        db = (out_features)
        global_grad = (B, T, out_features)

        dx += global_grad @ W # (B, T, out_features) @ (out_features, in_features) -> (B, T, in_features)
        dW += sum(global_grad.T @ x)  # (out_features, B * T) @ (B * T, in_features) -> (out_features, in_features)
        db += sum(global_grad)  # (B * T, out_features)
    */

    if (!linear->dW) 
        linear->dW = zeros(linear->W->shape, linear->W->ndims);

    if (!linear->db)
        linear->db = linear->use_bias > 0 ? zeros(linear->b->shape, linear->b->ndims) : NULL;

    tensor_t *dout, *ret;
    dout = zeros(linear->cache->shape, linear->cache->ndims);

    int B, T, out_features;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    out_features = global_grad->shape[2];

    // backprop into dx
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        B * T, linear->in_features, linear->out_features,
        1.0f, global_grad->t, linear->out_features, 
        linear->W->t, linear->in_features, 
        1.0f, dout->t, linear->in_features
    );

    // backprop into dW
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        linear->out_features, linear->in_features, B * T,
        1.0f, global_grad->t, linear->out_features, 
        linear->cache->t, linear->in_features, 
        1.0f, linear->dW->t, linear->in_features
    );

    // backprop into db
    if (linear->use_bias > 0) {
        int row_size = linear->db->shape[linear->db->ndims - 1];
        for (int i = 0; i < B * T; i++) {
            for (int j = 0; j < row_size; j++) {
                linear->db->t[j] += global_grad->t[i * row_size + j];
            }
        }
    }

    free_tensor(linear->cache);
    free_tensor(global_grad);
    linear->cache = NULL;
    global_grad = NULL;
    return dout;
}


void description_linear(const linear_t *linear) {
    if (linear == NULL)
        return;

    char w_shape[1024], b_shape[1024];
    shape(linear->W, w_shape);
    if (linear->use_bias > 0)
        shape(linear->b, b_shape);

    printf("Linear(in_features = %d, out_features = %d, use_bias = %d)\n", linear->in_features, linear->out_features, linear->use_bias);
    printf("----------------------------------------------------------\n");
    printf("  weight (%s): %d\n", w_shape, linear->W->length);
    if (linear->use_bias > 0)
    printf("  bias   (%s): %d\n", b_shape, linear->b->length);
    printf("\n");
}


int num_parameters_linear(const linear_t *linear) {
    if (linear == NULL)
        return 0;

    int total_parameters = 0;
    total_parameters += linear->W->length;
    
    if (linear->use_bias > 0)
        total_parameters += linear->b->length;

    return total_parameters;
}


void free_layer_linear(linear_t *linear) {
    if (linear == NULL) 
        return;

    free_tensor(linear->W);
    free_tensor(linear->b);
    free_tensor(linear->cache);
    free_tensor(linear->dW);
    free_tensor(linear->db);
    free(linear);
}


void free_cache_linear(linear_t *linear) {
    free_tensor(linear->cache);
    linear->cache = NULL;
}


tensor_t **parameters_linear(const linear_t * linear) {
    if (linear == NULL)
        return NULL;

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * linear->_num_param_tensors);
    parameters[0] = linear->W;
    if (linear->use_bias > 0)
        parameters[1] = linear->b;

    return parameters;
}


tensor_t **gradients_linear(const linear_t *linear) {
    if (linear == NULL)
        return NULL;

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * linear->_num_param_tensors);
    gradients[0] = linear->dW;
    if (linear->use_bias > 0)
        gradients[1] = linear->db;

    return gradients;
}


void load_state_dict_linear(linear_t *linear, tensor_t **state) {
    if (linear == NULL)
    {
        printf("Expected required arugment *linear to be of type linear_t ptr, but got NULL.\n");
        return;
    }

    if (state == NULL)
    {
        printf("Expected required argument **state to be of type tensor_t ** ptr, but got NULL.\n");
        return;
    }

    // check parameter and state length
    tensor_t *W = state[0];
    tensor_t *b = linear->use_bias > 0 ? state[0] : NULL;

    if (linear->W->length != W->length) {
        printf("Cannot load linear.weight as linear.W.length != state.W.length. Got %d != %d\n", linear->W->length, W->length);
        return;
    }

    if (linear->use_bias > 0 && linear->b->length != b->length) {
        printf("Cannot load linear.bias as linear.b.length != state.b.length. Got %d != %d\n", linear->b->length, b->length);
        return;
    }

    memcpy(linear->W->t, W->t, linear->W->length * sizeof(float));
    if (linear->use_bias > 0)
        memcpy(linear->b->t, b->t, linear->b->length * sizeof(float));
}
