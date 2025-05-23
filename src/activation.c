#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "activation.h"
#include "dispatch.h"

tensor_t *forward_gelu(gelu_t *gelu, tensor_t *x);
tensor_t *backward_gelu(gelu_t *gelu, tensor_t *global_grad);
void description_gelu(const gelu_t *gelu);
void free_layer_gelu(gelu_t *gelu);
void free_cache_gelu(gelu_t *gelu);
void to_gelu(gelu_t *gelu, const device_t device);


// GELU Class
gelu_t *GELU() {

    gelu_t *gelu = (gelu_t *)mallocCheck(sizeof(gelu_t));

    gelu->cache = NULL;
    gelu->forward = forward_gelu;
    gelu->backward = backward_gelu;
    gelu->description = description_gelu;
    gelu->free_layer = free_layer_gelu;
    gelu->free_cache = free_cache_gelu;
    gelu->to = to_gelu;
    return gelu;
}


tensor_t *forward_gelu(gelu_t *gelu, tensor_t *x) {
    CHECK_ERROR(gelu == NULL, "Expected *gelu to be a gelu_t pointer, but got NULL.");
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");

    tensor_t *out = create_tensor(x->shape, x->ndims, x->device);
    
    gelu_forward_dispatch(x, out);

    gelu->cache = x;
    return out;
}

tensor_t *backward_gelu(gelu_t *gelu, tensor_t *global_grad) {
    CHECK_ERROR(gelu == NULL, "Expected *gelu to be a gelu_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

    tensor_t *dout = zeros(gelu->cache->shape, gelu->cache->ndims, global_grad->device);

    gelu_backward_dispatch(global_grad, gelu->cache, dout);

    free_tensor(gelu->cache);
    free_tensor(global_grad);
    gelu->cache = NULL;
    global_grad = NULL;
    return dout;
}


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


void to_gelu(gelu_t *gelu, const device_t device) {
    CHECK_ERROR(gelu == NULL, "Expected *gelu to be a gelu_t pointer, but got NULL.");
    if (gelu->cache)
        gelu->cache->to(gelu->cache, device);
} 


tensor_t *forward_softmax(softmax_t *softmax, tensor_t *x);
tensor_t *backward_softmax(softmax_t *softmax, tensor_t *global_grad);
void description_softmax(const softmax_t *softmax);
void free_layer_softmax(softmax_t *softmax);
void free_cache_softmax(softmax_t *softmax);
void to_softmax(softmax_t *softmax, const device_t device);


// Softmax Class
softmax_t *Softmax() {

    softmax_t *softmax = (softmax_t *)mallocCheck(sizeof(softmax_t));

    softmax->cache = NULL;
    softmax->forward = forward_softmax;
    softmax->backward = backward_softmax;
    softmax->description = description_softmax;
    softmax->free_layer = free_layer_softmax;
    softmax->free_cache = free_cache_softmax;
    softmax->to = to_softmax;
    return softmax;
}


tensor_t *forward_softmax(softmax_t *softmax, tensor_t *x) {
    CHECK_ERROR(softmax == NULL, "Expected *softmax to be a softmax_t pointer, but got NULL.");
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");

    tensor_t *out = create_tensor(x->shape, x->ndims, x->device);

    softmax_forward_dispatch(x, out);

    softmax->cache = out;
    return out;
}


tensor_t *backward_softmax(softmax_t *softmax, tensor_t *global_grad) {
    CHECK_ERROR(softmax == NULL, "Expected *softmax to be a softmax_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

    tensor_t *dout = zeros(global_grad->shape, global_grad->ndims, global_grad->device);

    softmax_backward_dispatch(global_grad, softmax->cache, dout);

    free_tensor(softmax->cache);
    free_tensor(global_grad);
    softmax->cache = NULL;
    global_grad = NULL;
    return dout;
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


void to_softmax(softmax_t *softmax, const device_t device) {
    CHECK_ERROR(softmax == NULL, "Expected *softmax to be a softmax_t pointer, but got NULL.");
    if (softmax->cache)
        softmax->cache->to(softmax->cache, device);
}
