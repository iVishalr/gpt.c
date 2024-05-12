/**
 * \file            transformer.h
 * \brief           Header file for transformer
 */

/*
 * Copyright (c) 2024 Vishal Ramesha
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE
 * AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * This file is part of library_name.
 *
 * Author:          Vishal Ramesha
 */

#pragma once

#include "blocks.h"
#include "embedding.h"
#include "loss.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gpt2 {
    int n_embd;
    int n_heads;
    int n_layers;
    int block_size;
    int vocab_size;

    block_t **layers;

    embedding_t *wpe;
    embedding_t *wte;
    linear_t *lm_head;
    layer_norm_t *ln_f;

    tensor_t *(*forward)(struct gpt2 *, tensor_t *);
    tensor_t *(*backward)(struct gpt2 *, tensor_t *);

    void (*description)(const struct gpt2 *);
    int (*num_parameters)(const struct gpt2 *);
    void (*free_layer)(struct gpt2 *);

    tensor_t **(*parameters)(const struct gpt2 *);
    tensor_t **(*gradients)(const struct gpt2 *);
    void (*load_state_dict)(struct gpt2 *, tensor_t **);
    void (*fast_load_state_dict)(struct gpt2 *, tensor_t **);

    int _num_param_tensors;
} gpt2_t;

typedef struct {
    int block_size;
    int vocab_size;
    int n_layers;
    int n_heads;
    int n_embd;
} GPT2Config_t;

gpt2_t *GPT2(GPT2Config_t *config);

#ifdef __cplusplus
}
#endif