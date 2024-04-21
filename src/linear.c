#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

#include "linear.h"

void _calculate_fan_in_and_fan_out(tensor_t *t, int *result) {

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
    if (uniform(t, -bound, bound) == NULL) {
        printf("An error occured when initializing tensor.\n");
        return;
    }
}

tensor_t *forward_linear(linear_t *linear, const tensor_t *x) {
    
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
    
    int b_size = linear->b ? linear->b->shape[0] : 0;
    
    int collapsed_dims = 1;
    for (int i = 0; i < x->ndims - 1; i++)
        collapsed_dims *= x->shape[i];

    tensor_t *out, *ret;

    int out_shape[1024];
    for (int i = 0; i < x->ndims - 1; i++)
        out_shape[i] = x->shape[i];
    out_shape[x->ndims - 1] = linear->out_features;

    out = create_tensor(out_shape, x->ndims);

    ret = matmul(
        CblasRowMajor, CblasNoTrans, CblasTrans, 
        collapsed_dims, linear->out_features, linear->in_features, 
        1.0f, x, linear->in_features, 
        linear->W, linear->in_features, 
        0.0f, out, linear->out_features
    );

    if (ret == NULL) {
        printf("An error occured when performing matrix multiplication with shapes: [..., %d, %d] @ [%d, %d] -> [..., %d, %d]\n", collapsed_dims, linear->in_features, linear->in_features, linear->out_features, collapsed_dims, linear->out_features);
        free_tensor(out);
        out = NULL;
        return NULL;
    }

    // add bias to out tensor (B * T, out_features) + (out_features)
    if (linear->b) {
        int rows = 1;
        for (int i = 0; i < out->ndims - 1; i++)
            rows *= out->shape[i];
        
        for (int row = 0; row < rows; row++){
            for (int i = 0; i < b_size; i++) {
                out->t[row * b_size + i] += linear->b->t[i];
            }
        }
    }

    // cache the input to the layer
    linear->cache = create_tensor(x->shape, x->ndims);
    tensor_copy(linear->cache, x);
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

    if (linear->dW == NULL) 
        linear->dW = zeros(linear->W->shape, linear->W->ndims);

    if (linear->use_bias > 0 && linear->db == NULL) 
        linear->db = zeros(linear->b->shape, linear->b->ndims);

    // backprop into dx

    tensor_t *dout, *ret;
    dout = zeros(linear->cache->shape, linear->cache->ndims);

    int collapsed_dims = 1;
    for (int i = 0; i < dout->ndims - 1; i++)
        collapsed_dims *= dout->shape[i];

    ret = matmul(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        collapsed_dims, linear->in_features, linear->out_features,
        1.0f, global_grad, linear->out_features, 
        linear->W, linear->in_features, 
        1.0f, dout, linear->in_features
    );

    if (ret == NULL) {
        printf("An error occured when computing gradients towards input to the layer. [..., %d] @ [%d, %d] -> [..., %d]\n", linear->out_features, linear->out_features, linear->in_features, linear->in_features);
        free_tensor(dout);
        free_tensor(global_grad);
        free_tensor(linear->cache);
        dout = NULL;
        global_grad = NULL;
        linear->cache = NULL;
        return NULL;
    }

    // backprop into dW
    collapsed_dims = 1;
    for (int i = 0; i < global_grad->ndims - 1; i++)
        collapsed_dims *= global_grad->shape[i];

    ret = matmul(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        linear->out_features, linear->in_features, collapsed_dims,
        1.0f, global_grad, linear->out_features, 
        linear->cache, linear->in_features, 
        1.0f, linear->dW, linear->in_features
    );

    free_tensor(linear->cache);
    linear->cache = NULL;

    if (ret == NULL) {
        printf("An error occured when computing gradients towards weights to the layer. [%d, %d] @ [%d, %d] -> [%d, %d]\n", linear->out_features, collapsed_dims, collapsed_dims, linear->in_features, linear->out_features, linear->in_features);
        free_tensor(dout);
        free_tensor(global_grad);
        dout = NULL;
        global_grad = NULL;
        return NULL;
    }

    // backprop into db
    if (linear->use_bias > 0) {
        int db_size = linear->db->shape[linear->db->ndims - 1];
        for (int i = 0; i < collapsed_dims; i++) {
            for (int j = 0; j < db_size; j++) {
                linear->db->t[j] += global_grad->t[i * db_size + j];
            }
        }
    }

    free_tensor(global_grad);
    global_grad = NULL;
    return dout;
}

void description_linear(const linear_t *linear) {
    int parameters = linear->W->length;
    
    if (linear->b)
        parameters += linear->b->length;

    char w_shape[1024], b_shape[1024];
    shape(linear->W, w_shape);

    if (linear->use_bias > 0)
        shape(linear->b, b_shape);

    printf("Linear Layer\n");
    printf("------------\n");
    printf("in_features: %d\n", linear->in_features);
    printf("out_features: %d\n", linear->out_features);

    if (linear->use_bias > 0)
        printf("use_bias: True\n");
    else
        printf("use_bias: False\n");

    printf("num parameters: %d\n", parameters);
    printf("  W [%s]: %d\n", w_shape, linear->W->length);
    if (linear->use_bias > 0)
        printf("  b [%s]: %d\n", b_shape, linear->b->length);
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

linear_t *Linear(const int in_features, const int out_features, const int use_bias) {
    linear_t *linear = (linear_t *)malloc(sizeof(linear_t));
    int w_shape[2] = {out_features, in_features};
    tensor_t *W = zeros(w_shape, 2);
    tensor_t *b = NULL;
    if (use_bias > 0) {
        int b_shape[1] = {out_features};
        b = zeros(b_shape, 1);
    }

    // check if tensors were created
    if (W == NULL) {
        printf("An error occured when creating weight tensors.\n");
        return NULL;
    }

    if (use_bias > 0 && b == NULL) {
        printf("An error occured when creating bias tensors.\n");
        return NULL;
    }

    // init weights
    kaiming_uniform(W, sqrtf(5.0f), "fan_in", "leaky_relu");
    
    if (b != NULL) {
        // int *result = (int*)malloc(sizeof(int) * 2);
        int result[2];
        _calculate_fan_in_and_fan_out(W, result);
        int fan_in = result[0];
        float bound = fan_in > 0.0f ? 1.0f / sqrtf(fan_in): 0.0f;
        uniform(b, -bound, bound);
    }

    linear->W = W;
    linear->b = b;
    linear->in_features = in_features;
    linear->out_features = out_features;
    linear->use_bias = use_bias;
    linear->dW = NULL;
    linear->db = NULL;
    linear->cache = NULL;
    linear->forward = forward_linear;
    linear->backward = backward_linear;
    linear->description = description_linear;
    linear->free_layer = free_layer_linear;
    return linear;
}
