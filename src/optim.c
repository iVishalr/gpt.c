#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "utils.h"
#include "dispatch.h"
#include "optim.h"

#define ADAMW_DEFAULT_EPS 1e-08f
#define ADAMW_DEAFULT_WEIGHT_DECAY 0.01f


void step_adamW(adamW_t *optimizer);
void zero_grad_adamW(adamW_t *optimizer);
void free_layer_adamW(adamW_t *optimizer);


// AdamW Class
adamW_t *AdamW(tensor_t **parameters, tensor_t **gradients, const int n_parameters, const float lr, const float beta1, const float beta2, const float eps, const float weight_decay) {
    CHECK_ERROR(parameters == NULL, "Expected **parameters to be a tensor_t pointer, but got NULL.");
    CHECK_ERROR(gradients == NULL, "Expected **gradients to be a tensor_t pointer, but got NULL.");

    // verify parameters and gradients
    for (int i = 0; i < n_parameters; i++) {
        CHECK_ERROR(parameters[i] == NULL, "Expected parameters[%d] to be a tensor_t pointer, but got NULL.", i);
        CHECK_ERROR(gradients[i] == NULL, "Expected gradients[%d] to be a tensor_t pointer, but got NULL.", i);
        CHECK_ERROR(
            parameters[i]->ndims != gradients[i]->ndims, 
            "Expected parameters[%d] and gradients[%d] to be of same dimensions, but got %d != %d.", 
            i, i, parameters[i]->ndims, gradients[i]->ndims
        );
        CHECK_ERROR(
            parameters[i]->length != gradients[i]->length, 
            "Expected parameters[%d] and gradients[%d] to be of same length, but got %d != %d.",
            i, i, parameters[i]->length, gradients[i]->length
        );
    }

    adamW_t *optimizer = (adamW_t *)mallocCheck(sizeof(adamW_t));

    optimizer->parameters = parameters;
    optimizer->gradients = gradients;
    optimizer->n_parameters = n_parameters;
    optimizer->lr = lr;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->eps = eps != ADAMW_DEFAULT_EPS ? eps : ADAMW_DEFAULT_EPS;
    optimizer->weight_decay = weight_decay != ADAMW_DEAFULT_WEIGHT_DECAY ? weight_decay : ADAMW_DEAFULT_WEIGHT_DECAY;
    // optimizer->m = (tensor_t **)mallocCheck(sizeof(tensor_t *) * optimizer->n_parameters);
    // optimizer->v = (tensor_t **)mallocCheck(sizeof(tensor_t *) * optimizer->n_parameters);

    // for (int i = 0; i < optimizer->n_parameters; i++) {
    //     const tensor_t *grad = optimizer->gradients[i];
    //     const device_t device = grad->device;
    //     optimizer->m[i] = zeros(grad->shape, grad->ndims, device);
    //     optimizer->v[i] = zeros(grad->shape, grad->ndims, device);
    // }

    optimizer->m = NULL;
    optimizer->v = NULL;

    optimizer->step_t = 0;
    optimizer->step = step_adamW;
    optimizer->zero_grad = zero_grad_adamW;
    optimizer->free_layer = free_layer_adamW;
    return optimizer;
}


// Implemented as illustrated in https: // pytorch.org/docs/stable/generated/torch.optim.AdamW.html
void step_adamW(adamW_t *optimizer) {
    CHECK_ERROR(optimizer == NULL, "Expected *optimizer to be a adamW_t pointer, but got NULL.");

    // hate doing this here, but for some reason, moving this code to AdamW() causes 
    // a noticable slowdown in iteration speed on CPU :(.
    // This bug started when CPU objects were linked with CUDA objects.
    if (optimizer->m == NULL) {
        optimizer->m = (tensor_t **)mallocCheck(sizeof(tensor_t *) * optimizer->n_parameters);
        optimizer->v = (tensor_t **)mallocCheck(sizeof(tensor_t *) * optimizer->n_parameters);

        for (int i = 0; i < optimizer->n_parameters; i++)
        {
            const tensor_t *grad = optimizer->gradients[i];
            const device_t device = grad->device;
            optimizer->m[i] = zeros(grad->shape, grad->ndims, device);
            optimizer->v[i] = zeros(grad->shape, grad->ndims, device);
        }
    }

    optimizer->step_t += 1;

    step_adamW_dispatch(
        optimizer->parameters, 
        optimizer->gradients, 
        optimizer->m, 
        optimizer->v, 
        optimizer->n_parameters,
        optimizer->lr,
        optimizer->beta1,
        optimizer->beta2,
        optimizer->weight_decay,
        optimizer->eps,
        optimizer->step_t
    );

}


void zero_grad_adamW(adamW_t *optimizer) {
    CHECK_ERROR(optimizer == NULL, "Expected *optimizer to be a adamW_t pointer, but got NULL.");

    // set the gradients to 0
    zero_grad_adamW_dispatch(optimizer->gradients, optimizer->n_parameters);
}


void free_layer_adamW(adamW_t *optimizer) {
    if (optimizer == NULL)
        return;
        
    if (optimizer->m && optimizer->v) {
        for (int i = 0; i < optimizer->n_parameters; i++) {
            free_tensor(optimizer->m[i]);
            free_tensor(optimizer->v[i]);
        }
    }

    if (optimizer->m)
        free(optimizer->m);
    if (optimizer->v)
        free(optimizer->v);
    if (optimizer->parameters)
        free(optimizer->parameters);
    if (optimizer->gradients)
        free(optimizer->gradients);
    free(optimizer);
}
