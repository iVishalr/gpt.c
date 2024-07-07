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
    // Integer types grouped together
    int n_embd;             // 4 bytes
    int n_heads;            // 4 bytes
    int n_layers;           // 4 bytes
    int block_size;         // 4 bytes
    int vocab_size;         // 4 bytes
    int _num_param_tensors; // 4 bytes

    // Pointers grouped together
    block_t **layers;   // 8 bytes
    embedding_t *wpe;   // 8 bytes
    embedding_t *wte;   // 8 bytes
    linear_t *lm_head;  // 8 bytes
    layer_norm_t *ln_f; // 8 bytes

    // Function pointers grouped together
    tensor_t *(*forward)(struct gpt2 *, tensor_t *);          // 8 bytes
    tensor_t *(*backward)(struct gpt2 *, tensor_t *);         // 8 bytes
    void (*description)(const struct gpt2 *);                 // 8 bytes
    int (*num_parameters)(const struct gpt2 *);               // 8 bytes
    void (*free_layer)(struct gpt2 *);                        // 8 bytes
    void (*free_cache)(struct gpt2 *);                        // 8 bytes
    tensor_t **(*parameters)(const struct gpt2 *);            // 8 bytes
    tensor_t **(*gradients)(const struct gpt2 *);             // 8 bytes
    void (*load_state_dict)(struct gpt2 *, tensor_t **);      // 8 bytes
    void (*fast_load_state_dict)(struct gpt2 *, tensor_t **); // 8 bytes
} gpt2_t; // Ensure the structure is aligned to 8 bytes

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