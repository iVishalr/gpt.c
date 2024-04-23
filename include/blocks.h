/**
 * \file            blocks.h
 * \brief           Header file for creating various building blocks for transformer
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

#include "linear.h"
#include "attention.h"
#include "activation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlp {
    int in_features;
    int use_bias;
    int expansion_factor;
    
    linear_t *c_fc;
    gelu_t *gelu;
    linear_t *c_proj;
    
    tensor_t *(*forward)(struct mlp *, const tensor_t *);
    tensor_t *(*backward)(struct mlp *, tensor_t *);
    
    void (*description)(const struct mlp *);
    int (*num_parameters)(const struct mlp *);
    void (*free_layer)(struct mlp *);
} mlp_t;

typedef struct self_attention {
    int n_embd;
    int n_heads;
    int use_bias;
    int block_size;
    
    linear_t *qkv;
    attention_t *attn;
    linear_t *c_proj;

    tensor_t *(*forward)(struct mlp *, const tensor_t *);
    tensor_t *(*backward)(struct mlp *, tensor_t *);

    void (*description)(const struct self_attention *);
    int (*num_parameters)(const struct self_attention *);
    void (*free_layer)(struct self_attention *);
} self_attention_t;

mlp_t *MLP(const int in_features, int expansion_factor, const int use_bias);
self_attention_t *SelfAttention(const int n_embd, const int n_heads, const int block_size, const int use_bias);

#ifdef __cplusplus
}
#endif