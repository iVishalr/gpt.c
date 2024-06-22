/**
 * \file            embedding.h
 * \brief           Header file for Embedding layer
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

#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct embedding {
    // Pointers grouped together
    tensor_t *W;     // 8 bytes
    tensor_t *dW;    // 8 bytes
    tensor_t *cache; // 8 bytes

    // Function pointers grouped together
    tensor_t *(*forward)(struct embedding *, tensor_t *);     // 8 bytes
    tensor_t *(*backward)(struct embedding *, tensor_t *);    // 8 bytes
    void (*description)(const struct embedding *);            // 8 bytes
    int (*num_parameters)(const struct embedding *);          // 8 bytes
    void (*free_layer)(struct embedding *);                   // 8 bytes
    tensor_t **(*parameters)(const struct embedding *);       // 8 bytes
    tensor_t **(*gradients)(const struct embedding *);        // 8 bytes
    void (*load_state_dict)(struct embedding *, tensor_t **); // 8 bytes

    // Integers grouped together
    int num_embeddings;                    // 4 bytes
    int embedding_dim;                     // 4 bytes
    int _num_param_tensors;                // 4 bytes
    
    // Padding to maintain alignment of the entire struct
    char __padding[4];                     // 4 bytes
} __attribute__((aligned(64))) embedding_t; // Ensure the structure is aligned to 8 bytes

embedding_t *Embedding(int num_embeddings, int embedding_dim);

#ifdef __cplusplus
}
#endif