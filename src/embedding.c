#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "dispatch.h"
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
void to_embedding(embedding_t *embedding, const device_t device);


// Embedding Class
embedding_t *Embedding(int num_embeddings, int embedding_dim) {

    embedding_t *embedding = (embedding_t *)mallocCheck(sizeof(embedding_t));
    embedding->num_embeddings = num_embeddings;
    embedding->embedding_dim = embedding_dim;

    int wshape[2] = {num_embeddings, embedding_dim};
    embedding->W = randn(wshape, 2, CPU);
    embedding->dW = zeros(wshape, 2, CPU);
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
    embedding->to = to_embedding;
    embedding->_num_param_tensors = 1;
    return embedding;
}


tensor_t *forward_embedding(embedding_t *embedding, tensor_t *x) {

    CHECK_ERROR(embedding == NULL, "Expected *embedding to be a embedding_t pointer, but got NULL.");
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");

    /*
        out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
        x is (B,T) of integers, holding the token ids at each (b,t) position
        W is (V,C) of token embeddings, short for "weight token embeddings"

        for a given index "i", where i belongs to [0, num_embeddings] or [0, vocab_size],
        out[i] = W[i];
    */

    device_t device = x->device;
    int B, T, C;
    B = x->shape[0];
    T = x->shape[1];
    C = embedding->embedding_dim;
    
    int out_shape[3] = {B, T, C};
    tensor_t *out = create_tensor(out_shape, 3, device);

    embedding_forward_dispatch(embedding->W, x, out);

    embedding->cache = x;
    return out;
}


tensor_t *backward_embedding(embedding_t * embedding, tensor_t *global_grad) {

    CHECK_ERROR(embedding == NULL, "Expected *embedding to be a embedding_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

    device_t device = global_grad->device;
    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    if (!embedding->dW)
        embedding->dW = zeros(embedding->W->shape, embedding->W->ndims, device);

    embedding_backward_dispatch(global_grad, embedding->cache, embedding->dW);

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
    CHECK_ERROR(embedding == NULL, "Expected *embedding to be a embedding_t pointer, but got NULL.");
    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * embedding->_num_param_tensors);
    parameters[0] = embedding->W;
    return parameters;
}


tensor_t **gradients_embedding(const embedding_t *embedding) {
    CHECK_ERROR(embedding == NULL, "Expected *embedding to be a embedding_t pointer, but got NULL.");
    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * embedding->_num_param_tensors);
    gradients[0] = embedding->dW;
    return gradients;
}


void load_state_dict_embedding(embedding_t *embedding, tensor_t **state) {
    CHECK_ERROR(embedding == NULL, "Expected *embedding to be a embedding_t pointer, but got NULL.");
    CHECK_ERROR(state == NULL, "Expected **state to be a tensor_t pointer, but got NULL.");

    // check parameter and state length
    tensor_t *W = state[0];
    CHECK_ERROR(embedding->W->length != W->length, "Cannot load embedding weights. Expected a tensor of size %d, but %d", embedding->W->length, W->length);
    memcpy(embedding->W->t, W->t, embedding->W->length * sizeof(float));
}


void to_embedding(embedding_t *embedding, const device_t device) {
    CHECK_ERROR(embedding == NULL, "Expected *embedding to be a embedding_t pointer, but got NULL.");
    embedding->W->to(embedding->W, device);
    embedding->dW->to(embedding->dW, device);
    if (embedding->cache)
        embedding->cache->to(embedding->cache, device);
}