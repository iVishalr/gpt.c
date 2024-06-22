/**
 * \file            optim.h
 * \brief           Header file for Optimizer
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

typedef struct adamW {
    // Pointers grouped together
    tensor_t **parameters; // 8 bytes
    tensor_t **gradients;  // 8 bytes
    tensor_t **m;          // 8 bytes
    tensor_t **v;          // 8 bytes

    // Floats grouped together
    float lr;           // 4 bytes
    float beta1;        // 4 bytes
    float beta2;        // 4 bytes
    float eps;          // 4 bytes
    float weight_decay; // 4 bytes

    // Integers grouped together
    int n_parameters; // 4 bytes
    int step_t;       // 4 bytes
    
    // Padding to ensure 8-byte alignment of the entire structure
    char __padding[4]; // 4 bytes of padding

    // Function pointers grouped together
    void (*step)(struct adamW *);       // 8 bytes
    void (*zero_grad)(struct adamW *);  // 8 bytes
    void (*free_layer)(struct adamW *); // 8 bytes
} __attribute__((aligned(64))) adamW_t;  // Ensure the structure is aligned to 8 bytes

adamW_t *AdamW(tensor_t **parameters, tensor_t **gradients, const int n_parameters, const float lr, const float beta1, const float beta2, const float eps, const float weight_decay);

#ifdef __cplusplus
}
#endif