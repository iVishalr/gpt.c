#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "activation.h"

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)


tensor_t *forward_gelu(gelu_t *gelu, tensor_t *x);
tensor_t *backward_gelu(gelu_t *gelu, tensor_t *global_grad);
void description_gelu(const gelu_t *gelu);
void free_layer_gelu(gelu_t *gelu);
void free_cache_gelu(gelu_t *gelu);


// GELU Class
gelu_t *GELU() {

    gelu_t *gelu = (gelu_t *)mallocCheck(sizeof(gelu_t));

    gelu->cache = NULL;
    gelu->forward = forward_gelu;
    gelu->backward = backward_gelu;
    gelu->description = description_gelu;
    gelu->free_layer = free_layer_gelu;
    gelu->free_cache = free_cache_gelu;
    return gelu;
}


tensor_t *forward_gelu(gelu_t *gelu, tensor_t *x) {
    
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

    gelu->cache = x;
    return out;
}


// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it 
// (https://github.com/karpathy/llm.c/pull/200)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
tensor_t *backward_gelu(gelu_t *gelu, tensor_t *global_grad)
{
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
#pragma float_control(pop)


void description_gelu(const gelu_t *gelu) {
    printf("GELU()\n\n");
}


void free_layer_gelu(gelu_t *gelu) {
    if (gelu == NULL) 
        return;

    free_tensor(gelu->cache);
    free(gelu);
}


void free_cache_gelu(gelu_t *gelu) {
    if (gelu == NULL) 
        return;

    free_tensor(gelu->cache);
    gelu->cache = NULL;
}


tensor_t *forward_softmax(softmax_t *softmax, tensor_t *x);
tensor_t *backward_softmax(softmax_t *softmax, tensor_t *global_grad);
void description_softmax(const softmax_t *softmax);
void free_layer_softmax(softmax_t *softmax);
void free_cache_softmax(softmax_t *softmax);


// Softmax Class
softmax_t *Softmax()
{

    softmax_t *softmax = (softmax_t *)mallocCheck(sizeof(softmax_t));

    softmax->cache = NULL;
    softmax->forward = forward_softmax;
    softmax->backward = backward_softmax;
    softmax->description = description_softmax;
    softmax->free_layer = free_layer_softmax;
    softmax->free_cache = free_cache_softmax;
    return softmax;
}


tensor_t *forward_softmax(softmax_t *softmax, tensor_t *x) {
    
    if (softmax == NULL) {
        printf("Expected required arugment *softmax to be of type softmax_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    int B, T, C;
    B = x->shape[0];
    T = x->shape[1];
    C = x->shape[2];

    tensor_t *out = create_tensor(x->shape, x->ndims);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *x_bt = x->t + b * T * C + t * C;
            float *out_bt = out->t + b * T * C + t * C;

            float maxval = -INFINITY;
            for (int i = 0; i < C; i++)
                if (x_bt[i] > maxval)
                    maxval = x_bt[i];
            
            float sum = 0.0f;
            for (int i = 0; i < C; i++) {
                out_bt[i] = expf(x_bt[i] - maxval);
                sum += out_bt[i];
            }
            for (int i = 0; i < C; i++) {
                out_bt[i] /= sum;
            }
        }
    }
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

void free_cache_softmax(softmax_t *softmax) {
    if (softmax == NULL) 
        return;

    free_tensor(softmax->cache);
    softmax->cache = NULL;
}