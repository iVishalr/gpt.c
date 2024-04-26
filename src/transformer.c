#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include "transformer.h"

tensor_t *forward_transformer(gpt2_t *gpt, tensor_t *x, tensor_t *targets) {
    if (gpt == NULL) {
        printf("Expected required arugment *gpt to be of type gpt2_t ptr, but got NULL.\n");
        return NULL;
    }

    if (x == NULL) {
        printf("Expected required argument *x to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

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
    tensor_t *pos = create_tensor(pos_shape, 2);
    
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
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    block_t **layers;
    embedding_t *wpe, *wte;
    linear_t *lm_head;
    layer_norm_t *ln;

    wte = gpt->wte;
    wpe = gpt->wpe;
    lm_head = gpt->lm_head;
    layers = gpt->layers;
    ln = gpt->ln_f;

    int B, T, C;
    B = global_grad->shape[0];
    T = global_grad->shape[1];
    C = global_grad->shape[2];

    tensor_t *out = global_grad;
    out = lm_head->backward(lm_head, out);
    out = ln->backward(ln, out);

    for (int i = gpt->n_layers - 1; i >= 0; i--)
        out = layers[i]->backward(layers[i], out);

    int gg_pos_emb_shape[3] = {1, T, C};
    tensor_t *gg_pos_emb = zeros(gg_pos_emb_shape, 3);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *global_grad_bt = out->t + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                gg_pos_emb->t[t * C + i] += global_grad_bt[i];
            }
        }
    }

    tensor_t *d_pos_emb = wpe->backward(wpe, gg_pos_emb);
    tensor_t *d_tok_emb = wte->backward(wte, out);

    // add up gradients of wte.W (wte->dW) to lm_head->dW 
    // We need to do this because both the layers are sharing weights
    // wte.W = lm_head.W: (vocab_size, C)
    for (int v = 0; v < gpt->vocab_size; v++)
        cblas_saxpy(C, 1.0f, wte->dW->t + v * C, 1, lm_head->dW->t + v * C, 1);
    
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
    parameters += lm_head->num_parameters(lm_head);
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
    wte->W = NULL;
    wte->free_layer(wte);

    for (int i = 0; i < gpt->n_layers; i++)
        layers[i]->free_layer(layers[i]);

    ln->free_layer(ln);
    lm_head->free_layer(lm_head);

    free(layers);
    free(gpt);
}

gpt2_t *GPT2(GPT2Config_t *config) {
    gpt2_t *gpt = (gpt2_t *)malloc(sizeof(gpt2_t));

    gpt->block_size = config->block_size;
    gpt->vocab_size = config->vocab_size;
    gpt->n_embd = config->n_embd;
    gpt->n_heads = config->n_heads;
    gpt->n_layers = config->n_layers;

    gpt->wte = Embedding(gpt->vocab_size, gpt->n_embd);
    gpt->wpe = Embedding(gpt->block_size, gpt->n_embd);

    gpt->layers = (block_t **)malloc(gpt->n_layers * sizeof(block_t *));
    for (int i = 0; i < gpt->n_layers; i++)
        gpt->layers[i] = Block(gpt->n_embd, gpt->n_heads, gpt->block_size, 1);

    gpt->ln_f = LayerNorm(gpt->n_embd, 1e-5, 1);
    gpt->lm_head = Linear(gpt->n_embd, gpt->vocab_size, 0);

    free_tensor(gpt->wte->W);
    gpt->wte->W = gpt->lm_head->W; // https://paperswithcode.com/method/weight-tying

    gpt->forward = forward_transformer;
    gpt->backward = backward_transformer;
    gpt->free_layer = free_layer_transformer;
    gpt->description = description_transformer;
    gpt->num_parameters = num_parameters_transformer;
    return gpt;
}