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
#include "layer_norm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlp {
    // Integers grouped together
    int in_features;        // 4 bytes
    int use_bias;           // 4 bytes
    int expansion_factor;   // 4 bytes
    int _num_param_tensors; // 4 bytes

    // Pointers grouped together
    linear_t *c_fc;   // 8 bytes
    gelu_t *gelu;     // 8 bytes
    linear_t *c_proj; // 8 bytes

    // Function pointers grouped together
    tensor_t *(*forward)(struct mlp *, tensor_t *);     // 8 bytes
    tensor_t *(*backward)(struct mlp *, tensor_t *);    // 8 bytes
    void (*description)(const struct mlp *);            // 8 bytes
    int (*num_parameters)(const struct mlp *);          // 8 bytes
    void (*free_layer)(struct mlp *);                   // 8 bytes
    void (*free_cache)(struct mlp *);                   // 8 bytes
    tensor_t **(*parameters)(const struct mlp *);       // 8 bytes
    tensor_t **(*gradients)(const struct mlp *);        // 8 bytes
    void (*load_state_dict)(struct mlp *, tensor_t **); // 8 bytes
} __attribute__((aligned(64))) mlp_t;                    // Ensure the structure is aligned to 8 bytes

typedef struct self_attention {
    // Integers grouped together
    int n_embd;             // 4 bytes
    int n_heads;            // 4 bytes
    int use_bias;           // 4 bytes
    int block_size;         // 4 bytes
    int _num_param_tensors; // 4 bytes

    // Padding to maintain alignment of the entire struct
    char __padding[4]; // 4 bytes

    // Pointers grouped together
    linear_t *qkv;     // 8 bytes
    attention_t *attn; // 8 bytes
    linear_t *c_proj;  // 8 bytes

    // Function pointers grouped together
    tensor_t *(*forward)(struct self_attention *, tensor_t *);     // 8 bytes
    tensor_t *(*backward)(struct self_attention *, tensor_t *);    // 8 bytes
    void (*description)(const struct self_attention *);            // 8 bytes
    int (*num_parameters)(const struct self_attention *);          // 8 bytes
    void (*free_layer)(struct self_attention *);                   // 8 bytes
    void (*free_cache)(struct self_attention *);                   // 8 bytes
    tensor_t **(*parameters)(const struct self_attention *);       // 8 bytes
    tensor_t **(*gradients)(const struct self_attention *);        // 8 bytes
    void (*load_state_dict)(struct self_attention *, tensor_t **); // 8 bytes
} __attribute__((aligned(64))) self_attention_t;                    // Ensure the structure is aligned to 8 bytes

typedef struct block
{
    // Integers grouped together
    int n_embd;             // 4 bytes
    int n_heads;            // 4 bytes
    int use_bias;           // 4 bytes
    int block_size;         // 4 bytes
    int _num_param_tensors; // 4 bytes

    // Padding to maintain alignment of the entire struct
    char __padding[4]; // 4 bytes

    // Pointers grouped together
    layer_norm_t *ln1;      // 8 bytes
    layer_norm_t *ln2;      // 8 bytes
    mlp_t *mlp;             // 8 bytes
    self_attention_t *attn; // 8 bytes

    // Function pointers grouped together
    tensor_t *(*forward)(struct block *, tensor_t *);     // 8 bytes
    tensor_t *(*backward)(struct block *, tensor_t *);    // 8 bytes
    void (*description)(const struct block *);            // 8 bytes
    int (*num_parameters)(const struct block *);          // 8 bytes
    void (*free_layer)(struct block *);                   // 8 bytes
    void (*free_cache)(struct block *);                   // 8 bytes
    tensor_t **(*parameters)(const struct block *);       // 8 bytes
    tensor_t **(*gradients)(const struct block *);        // 8 bytes
    void (*load_state_dict)(struct block *, tensor_t **); // 8 bytes
} __attribute__((aligned(64))) block_t;                    // Ensure the structure is aligned to 8 bytes

mlp_t *MLP(const int in_features, int expansion_factor, const int use_bias);
self_attention_t *SelfAttention(const int n_embd, const int n_heads, const int block_size, const int use_bias);
block_t *Block(const int n_embd, const int n_heads, const int block_size, const int use_bias);

#ifdef __cplusplus
}
#endif