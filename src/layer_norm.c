#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "layer_norm.h"

#define DEFAULT_EPS 1e-5f

tensor_t *forward_layer_norm(layer_norm_t *norm, tensor_t *x) {
    
    if (norm == NULL) {
        printf("Expected required arugment *norm to be of type layer_norm_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = create_tensor(x->shape, x->ndims);
    int cache_shape[1024];
    for (int i = 0; i < x->ndims - 1; i++)
        cache_shape[i] = x->shape[i];

    norm->cache[0] = create_tensor(cache_shape, x->ndims - 1); // (B,T)
    norm->cache[1] = create_tensor(cache_shape, x->ndims - 1); // (B,T)

    int collapsed_dims = 1, row_size = x->shape[x->ndims - 1];
    for (int i = 0; i < x->ndims - 1; i++)
        collapsed_dims *= x->shape[i];

    for (int i = 0; i < collapsed_dims; i++) {
        float mean = 0.0f;
        for (int j = 0; j < row_size; j++) {
            mean += x->t[i * row_size + j];
        }
        mean = mean / row_size;

        // calculate variance
        float variance = 0.0f;
        for (int j = 0; j < row_size; j++) {
            float xshift = x->t[i * row_size + j] - mean;
            variance += xshift * xshift;
        }
        variance = variance / row_size;

        // calculate rstd (reciprocal standard deviation)
        float rstd = 1.0f / sqrtf(variance + norm->eps);

        for (int j = 0; j < row_size; j++) {
            float n = rstd * (x->t[i * row_size + j] - mean);
            float o = n * norm->W->t[j];
            if (norm->b)
                o += norm->b->t[j];
            out->t[i * row_size + j] = o;
        }

        norm->cache[0]->t[i] = mean;
        norm->cache[1]->t[i] = rstd;
    }

    if (x->requires_grad > 0) {
        norm->cache[2] = x;
    } else {
        norm->cache[2] = create_tensor(x->shape, x->ndims); // (B, T, C)
        tensor_copy(norm->cache[2], x);
    }
    return out;
}

tensor_t *backward_layer_norm(layer_norm_t *norm, tensor_t *global_grad) {

    if (norm == NULL) {
        printf("Expected required arugment *norm to be of type layer_norm_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

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

    tensor_t *mean, *rstd, *x, *out;
    mean = norm->cache[0];
    rstd = norm->cache[1];
    x = norm->cache[2];
    out = zeros(x->shape, x->ndims);
    norm->dW = zeros(norm->W->shape, norm->W->ndims);
    norm->db = norm->use_bias > 0 ? zeros(norm->b->shape, norm->b->ndims) : NULL;

    int collapsed_dims = 1;
    for (int i = 0; i < x->ndims - 1; i++) 
        collapsed_dims *= x->shape[i];

    int row_size = norm->in_features;

    for(int i = 0; i < collapsed_dims; i++) {
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int j = 0; j < row_size; j++) {
            float norm_i = (x->t[i * row_size + j] - mean->t[i]) * rstd->t[i];
            float dnorm_i = norm->W->t[j] * global_grad->t[i * row_size + j];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_i;
        }
        dnorm_mean = dnorm_mean / row_size;
        dnorm_norm_mean = dnorm_norm_mean / row_size;

        for (int j = 0; j < row_size; j++) {
            float norm_i = (x->t[i * row_size + j] - mean->t[i]) * rstd->t[i];
            float dnorm_i = norm->W->t[j] * global_grad->t[i * row_size + j];
            
            if (norm->db)
                norm->db->t[j] += global_grad->t[i * row_size + j];
            // gradient to weight
            norm->dW->t[j] += norm_i * global_grad->t[i * row_size + j];
            // gradient to input
            float dval = 0.0f;
            dval += dnorm_i;
            dval -= dnorm_mean;
            dval -= norm_i * dnorm_norm_mean;
            dval *= rstd->t[i];
            out->t[i * row_size + j] += dval;
        }
    }

    free_tensor(global_grad);
    free_tensor(norm->cache[0]);
    free_tensor(norm->cache[1]);
    free_tensor(norm->cache[2]);
    global_grad = NULL;
    norm->cache[0] = NULL;
    norm->cache[1] = NULL;
    norm->cache[2] = NULL;
    return out;
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

layer_norm_t *LayerNorm(int in_features, const float eps, const int use_bias) {
    
    if (in_features == 0) {
        printf("Expected in_features to be a value > 0, but got 0.\n");
        return NULL;
    }

    layer_norm_t *norm = (layer_norm_t *)malloc(sizeof(layer_norm_t));

    norm->cache[0] = NULL;
    norm->cache[1] = NULL;
    norm->cache[2] = NULL;
    norm->eps = eps != (float) DEFAULT_EPS ? eps : (float) DEFAULT_EPS;
    norm->use_bias = use_bias;
    norm->in_features = in_features;

    int param_shape[1] = {in_features};
    norm->W = ones(param_shape, 1);

    if (use_bias > 0)
        norm->b = zeros(param_shape, 1);
    else 
        norm->b = NULL;

    norm->dW = NULL;
    norm->db = NULL;

    norm->forward = forward_layer_norm;
    norm->backward = backward_layer_norm;
    norm->description = description_layer_norm;
    norm->num_parameters = num_parameters_layer_norm;
    norm->free_layer = free_layer_layer_norm;
    return norm;
}