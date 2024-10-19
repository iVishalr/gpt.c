#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "utils.h"
#include "optim.h"

#define ADAMW_DEFAULT_EPS 1e-08f
#define ADAMW_DEAFULT_WEIGHT_DECAY 0.01f


void step_adamW(adamW_t *optimizer);
void zero_grad_adamW(adamW_t *optimizer);
void free_layer_adamW(adamW_t *optimizer);


// AdamW Class
adamW_t *AdamW(tensor_t **parameters, tensor_t **gradients, const int n_parameters, const float lr, const float beta1, const float beta2, const float eps, const float weight_decay)
{
    if (parameters == NULL)
    {
        printf("Expected **parameters to be not NULL.\n");
        exit(EXIT_FAILURE);
    }

    if (gradients == NULL)
    {
        printf("Expected **gradients to be not NULL.\n");
        exit(EXIT_FAILURE);
    }

    // verify parameters and gradients
    for (int i = 0; i < n_parameters; i++)
    {
        if (parameters[i] == NULL)
        {
            printf("parameters contains a NULL ptr at position %d\n.", i);
            exit(EXIT_FAILURE);
        }

        if (gradients[i] == NULL)
        {
            printf("gradients contains a NULL ptr at position %d\n.", i);
            exit(EXIT_FAILURE);
        }

        if (parameters[i]->ndims != gradients[i]->ndims)
        {
            printf("Expected parameters and gradients at position %d to be of same dimensions. Got %d != %d\n", i, parameters[i]->ndims, gradients[i]->ndims);
            exit(EXIT_FAILURE);
        }

        if (parameters[i]->length != gradients[i]->length)
        {
            printf("Expected parameters and gradients at position %d to have same lengths. Got %d != %d\n", i, parameters[i]->length, gradients[i]->length);
            exit(EXIT_FAILURE);
        }
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
    if (optimizer == NULL)
        return;

    const float lr = optimizer->lr;
    const float weight_decay = optimizer->weight_decay;
    optimizer->step_t += 1;

    if (optimizer->m == NULL) {
        optimizer->m = (tensor_t **)mallocCheck(sizeof(tensor_t *) * optimizer->n_parameters);
        optimizer->v = (tensor_t **)mallocCheck(sizeof(tensor_t *) * optimizer->n_parameters);

        for (int i = 0; i < optimizer->n_parameters; i++) {
            tensor_t *grad;
            grad = optimizer->gradients[i];
            device_t device = grad->device;
            optimizer->m[i] = zeros(grad->shape, grad->ndims, device);
            optimizer->v[i] = zeros(grad->shape, grad->ndims, device);
        }
    }

    for (int i = 0; i < optimizer->n_parameters; i++) {
        tensor_t *param, *grad;
        param = optimizer->parameters[i];
        grad = optimizer->gradients[i];
        tensor_t *m_t = optimizer->m[i];
        tensor_t *v_t = optimizer->v[i];

        mul_(param, (1.0f - lr * weight_decay));
        
        mul_(m_t, optimizer->beta1);
        mul_(v_t, optimizer->beta2);
        
        float beta1_scale = (1.0f - optimizer->beta1);
        float beta2_scale = (1.0f - optimizer->beta2);
        
        for (int j = 0; j < grad->length; j++) {
            float grad_j = grad->t[j];
            m_t->t[j] += beta1_scale * grad_j;
            v_t->t[j] += beta2_scale * grad_j * grad_j;
        }

        beta1_scale = 1.0f / (1.0f - powf(optimizer->beta1, optimizer->step_t));
        beta2_scale = 1.0f / (1.0f - powf(optimizer->beta2, optimizer->step_t));

        float m_hat_j = 0.0f;
        float v_hat_j = 0.0f;
        for (int j = 0; j < grad->length; j++) {
            m_hat_j = m_t->t[j] * beta1_scale;
            v_hat_j = v_t->t[j] * beta2_scale;
            param->t[j] -= lr * m_hat_j / (sqrtf(v_hat_j) + optimizer->eps);
        }
    }   
}


void zero_grad_adamW(adamW_t *optimizer) {
    if (optimizer == NULL)
        exit(EXIT_FAILURE);

    // set the gradients to 0
    for (int i = 0; i < optimizer->n_parameters; i++) {
        tensor_t *grad = optimizer->gradients[i];
        for (int j = 0; j < grad->length; j++)
            grad->t[j] = 0.0f;
    }
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
