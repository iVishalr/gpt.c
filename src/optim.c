#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include "optim.h"

#define ADAMW_DEFAULT_EPS 1e-08f
#define ADAMW_DEAFULT_WEIGHT_DECAY 0.01f

// Implemented as illustrated in https: // pytorch.org/docs/stable/generated/torch.optim.AdamW.html
void step_adamW(adamW_t *optimizer) {
    if (optimizer == NULL)
        return;

    const float lr = optimizer->lr;
    const float weight_decay = optimizer->weight_decay;
    optimizer->step_t += 1;

    if (optimizer->m == NULL) {
        optimizer->m = (tensor_t **)malloc(sizeof(tensor_t *) * optimizer->n_parameters);
        optimizer->v = (tensor_t **)malloc(sizeof(tensor_t *) * optimizer->n_parameters);

        for (int i = 0; i < optimizer->n_parameters; i++) {
            tensor_t *grad;
            grad = optimizer->gradients[i];
            optimizer->m[i] = zeros(grad->shape, grad->ndims);
            optimizer->v[i] = zeros(grad->shape, grad->ndims);
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

        for (int j = 0; j < grad->length; j++) {
            float grad_j = grad->t[j];
            m_t->t[j] += (1.0f - optimizer->beta1) * grad_j;
            v_t->t[j] += (1.0f - optimizer->beta2) * grad_j * grad_j;
        }

        tensor_t *m_hat = create_tensor(m_t->shape, m_t->ndims);
        tensor_t *v_hat = create_tensor(v_t->shape, v_t->ndims);
        for (int j = 0; j < grad->length; j++) {
            m_hat->t[j] = m_t->t[j] / (1.0f - powf(optimizer->beta1, optimizer->step_t));
            v_hat->t[j] = v_t->t[j] / (1.0f - powf(optimizer->beta2, optimizer->step_t));
        }

        for (int j = 0; j < param->length; j++) {
            param->t[j] -= lr * m_hat->t[j] / (sqrtf(v_hat->t[j]) + optimizer->eps);
        }

        free_tensor(m_hat);
        free_tensor(v_hat);
    }   
}

void zero_grad_adamW(adamW_t *optimizer) {
    if (optimizer == NULL)
        return;

    // set the gradients to 0
    for (int i = 0; i < optimizer->n_parameters; i++) {
        tensor_t *grad = optimizer->gradients[i];
        for (int j = 0; j < grad->length; j++) {
            grad->t[j] = 0.0f;
        }
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

adamW_t *AdamW(tensor_t **parameters, tensor_t **gradients, const int n_parameters, const float lr, const float beta1, const float beta2, const float eps, const float weight_decay) {
    if (parameters == NULL) {
        printf("Expected **parameters to be not NULL.\n");
        return NULL;
    }

    if (gradients == NULL) {
        printf("Expected **gradients to be not NULL.\n");
        return NULL;
    }

    // verify parameters and gradients
    for (int i = 0; i < n_parameters; i++) {
        if (parameters[i] == NULL) {
            printf("parameters contains a NULL ptr at position %d\n.", i);
            return NULL;
        }

        if (gradients[i] == NULL) {
            printf("gradients contains a NULL ptr at position %d\n.", i);
            return NULL;
        }

        if (parameters[i]->ndims != gradients[i]->ndims) {
            printf("Expected parameters and gradients at position %d to be of same dimensions. Got %d != %d\n", i, parameters[i]->ndims, gradients[i]->ndims);
            return NULL;
        }

        if (parameters[i]->length != gradients[i]->length) {
            printf("Expected parameters and gradients at position %d to have same lengths. Got %d != %d\n", i, parameters[i]->length, gradients[i]->length);
            return NULL;
        }

    }

    adamW_t *optimizer = (adamW_t *)malloc(sizeof(adamW_t));

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