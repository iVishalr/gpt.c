/**
 * \file            layer_norm.h
 * \brief           Header file for LayerNorm layer
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

typedef struct layer_norm {
    // Integers grouped together
    int in_features;        // 4 bytes
    int use_bias;           // 4 bytes
    int _num_param_tensors; // 4 bytes

    // Floats grouped together
    float eps; // 4 bytes

    // Pointers grouped together
    tensor_t *W;        // 8 bytes
    tensor_t *b;        // 8 bytes
    tensor_t *dW;       // 8 bytes
    tensor_t *db;       // 8 bytes
    tensor_t *cache[3]; // 24 bytes (3 * 8 bytes)

    // Function pointers grouped together
    tensor_t *(*forward)(struct layer_norm *, tensor_t *);     // 8 bytes
    tensor_t *(*backward)(struct layer_norm *, tensor_t *);    // 8 bytes
    void (*description)(const struct layer_norm *);            // 8 bytes
    int (*num_parameters)(const struct layer_norm *);          // 8 bytes
    void (*free_layer)(struct layer_norm *);                   // 8 bytes
    void (*free_cache)(struct layer_norm *);                   // 8 bytes
    tensor_t **(*parameters)(const struct layer_norm *);       // 8 bytes
    tensor_t **(*gradients)(const struct layer_norm *);        // 8 bytes
    void (*load_state_dict)(struct layer_norm *, tensor_t **); // 8 bytes
    void (*to)(struct layer_norm *, const device_t);           // 8 bytes
} __attribute__((aligned(64))) layer_norm_t;                    // Ensure the structure is aligned to 8 bytes

layer_norm_t *LayerNorm(int in_features, const float eps, const int use_bias);

#ifdef __cplusplus
}
#endif