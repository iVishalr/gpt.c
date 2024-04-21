#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "activation.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

tensor_t *forward_gelu(gelu_t *gelu, const tensor_t *x) {
    
    if (gelu == NULL) {
        printf("Expected required arugment *gelu to be of type gelu_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = create_tensor(x->shape, x->ndims);

    for (int i = 0; i < x->length; i++) {
        float x_i = x->t[i];
        float cube = 0.044715f * x_i * x_i * x_i;
        out->t[i] = 0.5f * x_i * (1.0f + tanhf(GELU_SCALING_FACTOR * (x_i + cube)));
    }

    gelu->cache = create_tensor(x->shape, x->ndims);
    tensor_copy(gelu->cache, x);
    return out;
}

tensor_t *backward_gelu(gelu_t *gelu, tensor_t *global_grad) {
    if (gelu == NULL) {
        printf("Expected required arugment *gelu to be of type gelu_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *dout = zeros(gelu->cache->shape, gelu->cache->ndims);

    for (int i = 0; i < gelu->cache->length; i++) {
        float x_i = gelu->cache->t[i];
        float cube = 0.044715f * x_i * x_i * x_i;
        float tanh_arg = GELU_SCALING_FACTOR * (x_i + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x_i * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x_i * x_i);
        dout->t[i] += local_grad * global_grad->t[i];
    }

    free_tensor(gelu->cache);
    free_tensor(global_grad);
    gelu->cache = NULL;
    global_grad = NULL;

    return dout;
}

void description_gelu(const gelu_t *gelu) {
    printf("GELU()\n");
}

void free_layer_gelu(gelu_t *gelu) {
    if (gelu == NULL) 
        return;

    free_tensor(gelu->cache);
    free(gelu);
}

gelu_t *GELU() {
    
    gelu_t *gelu = (gelu_t*)malloc(sizeof(gelu_t));

    gelu->cache = NULL;
    gelu->forward = forward_gelu;
    gelu->backward = backward_gelu;
    gelu->description = description_gelu;
    gelu->free_layer = free_layer_gelu;

    return gelu;
}

tensor_t *forward_softmax(softmax_t *softmax, const tensor_t *x) {
    
    if (softmax == NULL) {
        printf("Expected required arugment *softmax to be of type softmax_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    tensor_t *out = create_tensor(x->shape, x->ndims);

    int collapsed_dims = 1;
    for (int i = 0; i < x->ndims - 1; i++) 
        collapsed_dims *= x->shape[i];

    int row_size = x->shape[x->ndims - 1];
    for (int row = 0; row < collapsed_dims; row++) {
        float max_val = -INFINITY;
        for (int j = 0; j < row_size; j++) {
            if (x->t[row * row_size + j] > max_val)
                max_val = x->t[row * row_size + j];
        }

        float sum = 0.0f, prob = 0.0f;
        for (int j = 0; j < row_size; j++) {
            out->t[row * row_size + j] = expf(x->t[row * row_size + j] - max_val);
            sum += out->t[row * row_size + j];
        }

        for (int j = 0; j < row_size; j++) {
            out->t[row * row_size + j] /= sum;
        }
    }

    // softmax->cache = create_tensor(x->shape, x->ndims);
    // tensor_copy(softmax->cache, x);
    return out;
}

tensor_t *backward_softmax(softmax_t *softmax, tensor_t *global_grad) {

    if (softmax == NULL) {
        printf("Expected required arugment *softmax to be of type gelu_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    printf("NotImplementedError: softmax.backward() is not implemented. Please use CrossEntropyLoss instead.\n");

    free_tensor(softmax->cache);
    free_tensor(global_grad);
    softmax->cache = NULL;
    global_grad = NULL;
    return NULL;
}

void description_softmax(const softmax_t *softmax) {
    printf("Softmax()\n");
}

void free_layer_softmax(softmax_t *softmax) {
    if (softmax == NULL) 
        return;

    free_tensor(softmax->cache);
    free(softmax);
}

softmax_t *Softmax() {
    
    softmax_t *softmax = (softmax_t *)malloc(sizeof(softmax_t));

    softmax->cache = NULL;
    softmax->forward = forward_softmax;
    softmax->backward = backward_softmax;
    softmax->description = description_softmax;
    softmax->free_layer = free_layer_softmax;

    return softmax;
}
