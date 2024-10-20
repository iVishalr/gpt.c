#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "utils.h"
#include "transformer.h"


tensor_t *forward_transformer(gpt2_t *gpt, tensor_t *x);
tensor_t *backward_transformer(gpt2_t *gpt, tensor_t *global_grad);
int num_parameters_transformer(const gpt2_t *gpt);
void description_transformer(const gpt2_t *gpt);
void free_layer_transformer(gpt2_t *gpt);
void free_cache_transformer(gpt2_t *gpt);
tensor_t **parameters_transformer(const gpt2_t *gpt);
tensor_t **gradients_transformer(const gpt2_t *gpt);
void load_state_dict_transformer(gpt2_t *gpt, tensor_t **state);
void fast_load_state_dict_transformer(gpt2_t *gpt, tensor_t **state);
void to_transformer(gpt2_t *gpt, const device_t device);


// GPT2 Class
gpt2_t *GPT2(GPT2Config_t *config) {
    gpt2_t *gpt = (gpt2_t *)mallocCheck(sizeof(gpt2_t));

    gpt->block_size = config->block_size;
    gpt->vocab_size = config->vocab_size;
    gpt->n_embd = config->n_embd;
    gpt->n_heads = config->n_heads;
    gpt->n_layers = config->n_layers;

    gpt->wte = Embedding(gpt->vocab_size, gpt->n_embd);
    gpt->wpe = Embedding(gpt->block_size, gpt->n_embd);

    gpt->layers = (block_t **)mallocCheck(gpt->n_layers * sizeof(block_t *));
    for (int i = 0; i < gpt->n_layers; i++)
        gpt->layers[i] = Block(gpt->n_embd, gpt->n_heads, gpt->block_size, 1);

    gpt->ln_f = LayerNorm(gpt->n_embd, 1e-5, 1);
    gpt->lm_head = Linear(gpt->n_embd, gpt->vocab_size, 0);

    free_tensor(gpt->lm_head->W);
    free_tensor(gpt->lm_head->dW);
    gpt->lm_head->W = gpt->wte->W; // https://paperswithcode.com/method/weight-tying
    gpt->lm_head->dW = NULL;

    gpt->forward = forward_transformer;
    gpt->backward = backward_transformer;
    gpt->free_layer = free_layer_transformer;
    gpt->free_cache = free_cache_transformer;
    gpt->description = description_transformer;
    gpt->num_parameters = num_parameters_transformer;
    gpt->parameters = parameters_transformer;
    gpt->gradients = gradients_transformer;
    gpt->load_state_dict = load_state_dict_transformer;
    gpt->fast_load_state_dict = fast_load_state_dict_transformer;
    gpt->to = to_transformer;

    gpt->_num_param_tensors = gpt->wpe->_num_param_tensors;
    for (int i = 0; i < gpt->n_layers; i++)
        gpt->_num_param_tensors += gpt->layers[i]->_num_param_tensors;
    gpt->_num_param_tensors += gpt->ln_f->_num_param_tensors;
    gpt->_num_param_tensors += gpt->lm_head->_num_param_tensors;

    return gpt;
}


tensor_t *forward_transformer(gpt2_t *gpt, tensor_t *x) {
    if (gpt == NULL) {
        printf("Expected required arugment *gpt to be of type gpt2_t ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    device_t device = x->device;
    int B, T, C;
    B = x->shape[0];
    T = x->shape[1];
    C = gpt->n_embd;

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wte = gpt->wte;
    wpe = gpt->wpe;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    tensor_t *tok_emb = wte->forward(wte, x);

    int pos_shape[2] = {1, T};
    tensor_t *pos = create_tensor(pos_shape, 2, device);
    
    for (int t = 0; t < T; t++)
        pos->t[t] = t;

    tensor_t *pos_emb = wpe->forward(wpe, pos);

    for (int b = 0; b < B; b++)
        cblas_saxpy(T * C, 1.0f, pos_emb->t, 1, tok_emb->t + b * T * C, 1);

    free_tensor(pos_emb);
    pos_emb = NULL;
    
    tensor_t *out = tok_emb;

    for (int i = 0; i < gpt->n_layers; i++)
        out = layers[i]->forward(layers[i], out);

    out = ln->forward(ln, out);
    out = lm_head->forward(lm_head, out);
    return out;
}


tensor_t *backward_transformer(gpt2_t *gpt, tensor_t *global_grad) {
    if (gpt == NULL) {
        printf("Expected required arugment *gpt to be of type gpt2_t ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    device_t device = global_grad->device;
    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wte = gpt->wte;
    wpe = gpt->wpe;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    int B, T, C, n_embd;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];
    n_embd = gpt->n_embd;

    tensor_t *out = global_grad;
    
    free_tensor(lm_head->dW);
    lm_head->dW = NULL;

    out = lm_head->backward(lm_head, out);
    out = ln->backward(ln, out);
    
    for (int i = gpt->n_layers - 1; i >= 0; i--)
        out = layers[i]->backward(layers[i], out);

    int gg_pos_emb_shape[3] = {1, T, n_embd};
    tensor_t *gg_pos_emb = zeros(gg_pos_emb_shape, 3, device);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *global_grad_bt = out->t + b * T * n_embd + t * n_embd;
            for (int i = 0; i < n_embd; i++) {
                gg_pos_emb->t[t * n_embd + i] += global_grad_bt[i];
            }
        }
    }

    tensor_t *d_pos_emb = wpe->backward(wpe, gg_pos_emb);
    tensor_t *d_tok_emb = wte->backward(wte, out);

    // add up gradients of wte.W (wte->dW) to lm_head->dW 
    // We need to do this because both the layers are sharing weights
    // wte.W = lm_head.W: (vocab_size, C)
    for (int v = 0; v < gpt->vocab_size; v++)
        cblas_saxpy(n_embd, 1.0f, lm_head->dW->t + v * n_embd, 1, wte->dW->t + v * n_embd, 1);
    
    return d_tok_emb;
}


int num_parameters_transformer(const gpt2_t *gpt) {
    if (gpt == NULL)
        return 0;

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    int parameters = 0;
    parameters += wpe->num_parameters(wpe);
    parameters += wte->num_parameters(wte);

    for (int i = 0; i < gpt->n_layers; i++)
        parameters += layers[i]->num_parameters(layers[i]);

    parameters += ln->num_parameters(ln);
    return parameters;
}


void description_transformer(const gpt2_t *gpt) {
    if (gpt == NULL)
        return;

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    int parameters = gpt->num_parameters(gpt);

    printf("[GPT2]\n");
    printf("vocab_size: %d\n", gpt->vocab_size);
    printf("block_size: %d\n", gpt->block_size);
    printf("n_embd    : %d\n", gpt->n_embd);
    printf("n_heads   : %d\n", gpt->n_heads);
    printf("n_layers  : %d\n\n", gpt->n_layers);

    wte->description(wte);
    wpe->description(wpe);
    
    for (int i = 0; i < gpt->n_layers; i++)
        gpt->layers[i]->description(gpt->layers[i]);

    ln->description(ln);
    lm_head->description(lm_head);

    printf("-------------------------------\n");
    printf("Parameters: %d\n", parameters);
    printf("Memory Used: %lf MB\n", ((double) parameters * sizeof(float) / 1024 / 1024));
    printf("-------------------------------\n");
}


void free_layer_transformer(gpt2_t *gpt) {
    if (gpt == NULL)
        return;

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    wpe->free_layer(wpe);
    wte->free_layer(wte);

    for (int i = 0; i < gpt->n_layers; i++)
        layers[i]->free_layer(layers[i]);

    ln->free_layer(ln);
    lm_head->W = NULL;
    lm_head->free_layer(lm_head);

    free(layers);
    free(gpt);
}


void free_cache_transformer(gpt2_t *gpt) {
    if (gpt == NULL)
        return;

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    wpe->free_cache(wpe);
    wte->free_cache(wte);

    for (int i = 0; i < gpt->n_layers; i++)
        layers[i]->free_cache(layers[i]);

    ln->free_cache(ln);
    lm_head->free_cache(lm_head);
}


tensor_t **parameters_transformer(const gpt2_t *gpt) {
    if (gpt == NULL)
        exit(EXIT_FAILURE);

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    tensor_t **parameters = (tensor_t **)mallocCheck(sizeof(tensor_t *) * gpt->_num_param_tensors);

    int idx = 0;

    tensor_t **wte_params = wte->parameters(wte);
    for (int i = 0; i < wte->_num_param_tensors; i++)
        parameters[idx++] = wte_params[i];

    free(wte_params);

    tensor_t **wpe_params = wpe->parameters(wpe);
    for (int i = 0; i < wpe->_num_param_tensors; i++)
        parameters[idx++] = wpe_params[i];

    free(wpe_params);

    for (int i = 0; i < gpt->n_layers; i++) {
        tensor_t **layer_params = layers[i]->parameters(layers[i]);
        for (int j = 0; j < layers[i]->_num_param_tensors; j++)
            parameters[idx++] = layer_params[j];
        free(layer_params);
    }

    tensor_t **ln_params = ln->parameters(ln);
    for (int i = 0; i < ln->_num_param_tensors; i++)
        parameters[idx++] = ln_params[i];

    free(ln_params);

    return parameters;
}


tensor_t **gradients_transformer(const gpt2_t *gpt) {
    if (gpt == NULL)
        exit(EXIT_FAILURE);

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    tensor_t **gradients = (tensor_t **)mallocCheck(sizeof(tensor_t *) * gpt->_num_param_tensors);

    int idx = 0;

    tensor_t **wte_grads = wte->gradients(wte);
    for (int i = 0; i < wte->_num_param_tensors; i++)
        gradients[idx++] = wte_grads[i];

    free(wte_grads);

    tensor_t **wpe_grads = wpe->gradients(wpe);
    for (int i = 0; i < wpe->_num_param_tensors; i++)
        gradients[idx++] = wpe_grads[i];

    free(wpe_grads);


    for (int i = 0; i < gpt->n_layers; i++) {
        tensor_t **layer_grads = layers[i]->gradients(layers[i]);
        for (int j = 0; j < layers[i]->_num_param_tensors; j++)
            gradients[idx++] = layer_grads[j];
        free(layer_grads);
    }

    tensor_t **ln_grads = ln->gradients(ln);
    for (int i = 0; i < ln->_num_param_tensors; i++)
        gradients[idx++] = ln_grads[i];

    free(ln_grads);
    return gradients;
}


void load_state_dict_transformer(gpt2_t *gpt, tensor_t **state) {
    if (gpt == NULL) {
        printf("Expected required arugment *gpt to be of type gpt2_t ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    if (state == NULL) {
        printf("Expected required argument **state to be of type tensor_t ** ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    wte->load_state_dict(wte, state);
    state += wte->_num_param_tensors;
    
    wpe->load_state_dict(wpe, state);
    state += wpe->_num_param_tensors;

    for (int i = 0; i < gpt->n_layers; i++) {
        block_t *layer = layers[i];
        layer->load_state_dict(layer, state);
        state += layer->_num_param_tensors;
    }

    ln->load_state_dict(ln, state);
    state += ln->_num_param_tensors;
    return;
}


void fast_load_state_dict_transformer(gpt2_t *gpt, tensor_t **state) {
    if (gpt == NULL) {
        printf("Expected required arugment *gpt to be of type gpt2_t ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    if (state == NULL) {
        printf("Expected required argument **state to be of type tensor_t ** ptr, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    tensor_t **parameters = gpt->parameters(gpt);
    for (int i = 0; i < gpt->_num_param_tensors; i++) {
        tensor_t *model_param = parameters[i];
        tensor_t *state_param = state[i];

        if (model_param->length != state_param->length) {
            char a_shape[1024], b_shape[1024];
            shape(model_param, a_shape);
            shape(state_param, b_shape);
            printf("Expected both model parameters and state tensor length's to match at index %d. Got (%s) != (%s)\n", i, a_shape, b_shape);
            exit(1);
        }

        memcpy(model_param->t, state_param->t, model_param->length * sizeof(float));
    }
    free(parameters);
}


void to_transformer(gpt2_t *gpt, const device_t device) {
    if (gpt == NULL) {
        printf("Expected required arugment *gpt to be of type gpt2_t ptr, but got NULL.\n");
        return;
    }

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wpe = gpt->wpe;
    wte = gpt->wte;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    wpe->to(wpe, device);
    wte->to(wte, device);

    for (int i = 0; i < gpt->n_layers; i++)
        layers[i]->to(layers[i], device);

    ln->to(ln, device);
    lm_head->to(lm_head, device);
}
