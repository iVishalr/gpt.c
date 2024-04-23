#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "embedding.h"

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

    int out_shape[1024];
    int index = 0;
    int collapsed_dims = 1;
    
    for (int i = 0; i < x->ndims; i++) {
        out_shape[index++] = x->shape[i];
        collapsed_dims *= x->shape[i];
    }
    collapsed_dims /= x->shape[x->ndims - 1];
    out_shape[index] = embedding->embedding_dim;
    
    tensor_t *out = zeros(out_shape, index + 1);
    out->requires_grad = 1;

    int row_size = embedding->embedding_dim;
    for (int i = 0; i < collapsed_dims; i++) {
        for (int t = 0; t < x->shape[1]; t++) {
            int ix = (int)x->t[i * x->shape[1] + t];
            for (int j = 0; j < row_size; j++) {
                out->t[i * (x->shape[1] * row_size) + t * row_size + j] = embedding->W->t[ix * row_size + j];
            }
        }
    }

    if (x->requires_grad > 0) {
        embedding->cache = x;
    } else {
        embedding->cache = create_tensor(x->shape, x->ndims);
        tensor_copy(embedding->cache, x);
    }
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
    
    embedding->dW = zeros(embedding->W->shape, embedding->W->ndims);

    tensor_t *out = zeros(embedding->cache->shape, embedding->cache->ndims);

    for (int i = 0; i < global_grad->shape[0]; i++) {
        for (int t = 0; t < global_grad->shape[1]; t++) {
            int ix = (int)embedding->cache->t[i * global_grad->shape[0] + t];
            for (int j = 0; j < embedding->W->shape[1]; j++) {
                embedding->dW->t[ix * embedding->embedding_dim + j] += global_grad->t[i * (global_grad->shape[1] * embedding->embedding_dim) + t * embedding->embedding_dim + j];
            }
        }
    }

    free_tensor(global_grad);
    free_tensor(embedding->cache);
    embedding->cache = NULL;
    global_grad = NULL;
    return out;
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
    printf("  W [%s]: %d\n", wshape, embedding->W->length);
}

void free_layer_embedding(embedding_t *embedding) {
    if (embedding == NULL) 
        return;

    free_tensor(embedding->W);
    free_tensor(embedding->dW);
    free_tensor(embedding->cache);
    free(embedding);
}

embedding_t *Embedding(int num_embeddings, int embedding_dim) {

    embedding_t *embedding = (embedding_t *)malloc(sizeof(embedding_t));
    embedding->num_embeddings = num_embeddings;
    embedding->embedding_dim = embedding_dim;
    
    int wshape[2] = {num_embeddings, embedding_dim};

    embedding->W = randn(wshape, 2);
    embedding->dW = NULL;
    embedding->cache = NULL;
    embedding->forward = forward_embedding;
    embedding->backward = backward_embedding;
    embedding->description = description_embedding;
    embedding->num_parameters = num_parameters_embedding;
    embedding->free_layer = free_layer_embedding;
    return embedding;
}