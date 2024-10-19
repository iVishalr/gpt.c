/**
 * \file            linear.h
 * \brief           Header file for Linear layer
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

typedef struct linear {
    // Pointers grouped together (8 bytes each on a 64-bit system)
    tensor_t *W;     // 8 bytes
    tensor_t *b;     // 8 bytes
    tensor_t *dW;    // 8 bytes
    tensor_t *db;    // 8 bytes
    tensor_t *cache; // 8 bytes

    // Function pointers grouped together (8 bytes each on a 64-bit system)
    tensor_t *(*forward)(struct linear *, tensor_t *);     // 8 bytes
    tensor_t *(*backward)(struct linear *, tensor_t *);    // 8 bytes
    void (*description)(const struct linear *);            // 8 bytes
    int (*num_parameters)(const struct linear *);          // 8 bytes
    void (*free_layer)(struct linear *);                   // 8 bytes
    void (*free_cache)(struct linear *);                   // 8 bytes
    tensor_t **(*parameters)(const struct linear *);       // 8 bytes
    tensor_t **(*gradients)(const struct linear *);        // 8 bytes
    void (*load_state_dict)(struct linear *, tensor_t **); // 8 bytes
    void (*to)(struct linear *, const device_t);           // 8 bytes

    // Integers grouped together (4 bytes each)
    int in_features;        // 4 bytes
    int out_features;       // 4 bytes
    int use_bias;           // 4 bytes
    int _num_param_tensors; // 4 bytes

    // Padding to maintain alignment of the entire struct
    // char __padding[8]; // 8 bytes to align the struct size to a multiple of 8 bytes
} __attribute__((aligned(64))) linear_t;

linear_t *Linear(const int in_features, const int out_features, const int use_bias);

#ifdef __cplusplus
}
#endif