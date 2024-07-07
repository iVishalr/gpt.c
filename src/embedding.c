#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "embedding.h"


tensor_t *forward_embedding(embedding_t *embedding, tensor_t *x);
tensor_t *backward_embedding(embedding_t *embedding, tensor_t *global_grad);
int num_parameters_embedding(const embedding_t *embedding);
void description_embedding(const embedding_t *embedding);
void free_layer_embedding(embedding_t *embedding);
void free_cache_embedding(embedding_t *embedding);
tensor_t **parameters_embedding(const embedding_t *embedding);
tensor_t **gradients_embedding(const embedding_t *embedding);
void load_state_dict_embedding(embedding_t *embedding, tensor_t **state);


// Embedding Class
embedding_t *Embedding(int num_embeddings, int embedding_dim) {

    embedding_t *embedding = (embedding_t *)mallocCheck(sizeof(embedding_t));
    embedding->num_embeddings = num_embeddings;
    embedding->embedding_dim = embedding_dim;

    int wshape[2] = {num_embeddings, embedding_dim};
    embedding->W = randn(wshape, 2);
    embedding->dW = zeros(wshape, 2);
    embedding->cache = NULL;
    embedding->forward = forward_embedding;
    embedding->backward = backward_embedding;
    embedding->description = description_embedding;
    embedding->num_parameters = num_parameters_embedding;
    embedding->free_layer = free_layer_embedding;
    embedding->free_cache = free_cache_embedding;
    embedding->parameters = parameters_embedding;
    embedding->gradients = gradients_embedding;
    embedding->load_state_dict = load_state_dict_embedding;
    embedding->_num_param_tensors = 1;
    return embedding;
}

tensor_t *forward_embedding(embedding_t *embedding, tensor_t *x) {
    
    if (embedding == NULL) {
        printf("Expected required arugment *embedding to be of type embedding_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    /*
        out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        x is (B,T) of integers, holding the token ids at each (b,t) position
        W is (V,C) of token embeddings, short for "weight token embeddings"

        for a given index "i", where i belongs to [0, num_embeddings] or [0, vocab_size],
        out[i] = W[i];
    */

    int B, T, C;
    B = x->shape[0];
    T = x->shape[1];
    C = embedding->embedding_dim;
    
    int out_shape[3] = {B, T, C};
    tensor_t *out = zeros(out_shape, 3);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = out->t + b * T * C + t * C;
            int ix = (int)x->t[b * T + t];
            float *w_ix = embedding->W->t + ix * C;
            for (int i = 0; i < C; i++)
                out_bt[i] = w_ix[i];
        }
    }
    
    embedding->cache = x;
    return out;
}

tensor_t *backward_embedding(embedding_t * embedding, tensor_t *global_grad) {
    
    if (embedding == NULL) {
        printf("Expected required arugment *embedding to be of type embedding_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }
    
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    if (!embedding->dW)
        embedding->dW = zeros(embedding->W->shape, embedding->W->ndims);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *global_grad_bt = global_grad->t + b * T * C + t * C;
            int ix = (int)embedding->cache->t[b * T + t];
            float *dW_ix = embedding->dW->t + ix * C;
            for (int i = 0; i < C; i++) {
                dW_ix[i] += global_grad_bt[i];
            }
        }
    }

    free_tensor(global_grad);
    free_tensor(embedding->cache);
    embedding->cache = NULL;
    global_grad = NULL;
    return NULL;
}


int num_parameters_embedding(const embedding_t *embedding) {
    if (embedding == NULL)
        return 0;

    int total_parameters = embedding->W->length;
    return total_parameters;
}


void description_embedding(const embedding_t *embedding) {
    if (embedding == NULL)
        return;

    int parameters = embedding->W->length;
    char wshape[1024];
    shape(embedding->W, wshape);
    printf("Embedding\n");
    printf("---------\n");
    printf("num_embeddings: %d\n", embedding->num_embeddings);
    printf("embedding_dim: %d\n", embedding->embedding_dim);
    printf("num parameters: %d\n", parameters);
    printf("  W [%s]: %d\n\n", wshape, embedding->W->length);
}


void free_layer_embedding(embedding_t *embedding) {
    if (embedding == NULL) 
        return;

    free_tensor(embedding->W);
    free_tensor(embedding->dW);
    free_tensor(embedding->cache);
    free(embedding);
}


void free_cache_embedding(embedding_t *embedding) {
    if (embedding == NULL)
        return;

    free_tensor(embedding->cache);
    embedding->cache = NULL;
}


tensor_t **parameters_embedding(const embedding_t *embedding) {
    if (embedding == NULL)
        return NULL;

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * embedding->_num_param_tensors);
    parameters[0] = embedding->W;
    return parameters;
}


tensor_t **gradients_embedding(const embedding_t *embedding) {
    if (embedding == NULL)
        return NULL;

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * embedding->_num_param_tensors);
    gradients[0] = embedding->dW;
    return gradients;
}


void load_state_dict_embedding(embedding_t *embedding, tensor_t **state)
{
    if (embedding == NULL)
    {
        printf("Expected required arugment *embedding to be of type embedding_t ptr, but got NULL.\n");
        return;
    }

    if (state == NULL)
    {
        printf("Expected required argument **state to be of type tensor_t ** ptr, but got NULL.\n");
        return;
    }

    // check parameter and state length
    tensor_t *W = state[0];

    if (embedding->W->length != W->length)
    {
        printf("Cannot load embedding.weight as embedding.W.length != state.W.length. Got %d != %d\n", embedding->W->length, W->length);
        return;
    }

    memcpy(embedding->W->t, W->t, embedding->W->length * sizeof(float));
}
