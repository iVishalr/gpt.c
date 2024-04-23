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

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct embedding {
    tensor_t *W;
    tensor_t *dW;
    tensor_t *cache;
    tensor_t *(*forward)(struct embedding *, tensor_t *);
    tensor_t *(*backward)(struct embedding *, tensor_t *);
    void (*description)(const struct embedding *);
    int (*num_parameters)(const struct embedding *);
    void (*free_layer)(struct embedding *);
    int num_embeddings;
    int embedding_dim;
} embedding_t;

embedding_t *Embedding(int num_embeddings, int embedding_dim);

#ifdef __cplusplus
}
#endif