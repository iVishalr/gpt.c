/**
 * \file            attention.h
 * \brief           Header file for Attention layer
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

typedef struct attention {
    // Pointers grouped together
    tensor_t *buffer;   // 8 bytes
    tensor_t *cache[4]; // 4 * 8 bytes = 32 bytes (assuming pointers are 8 bytes each)

    // Function pointers grouped together
    tensor_t *(*forward)(struct attention *, tensor_t *);  // 8 bytes
    tensor_t *(*backward)(struct attention *, tensor_t *); // 8 bytes
    void (*description)(const struct attention *);         // 8 bytes
    int (*num_parameters)(const struct attention *);       // 8 bytes
    void (*free_layer)(struct attention *);                // 8 bytes
    void (*free_cache)(struct attention *);                // 8 bytes
    void (*to)(struct attention *, const device_t device); // 8 bytes

    // Integers grouped together
    int n_embd;   // 4 bytes
    int n_heads;  // 4 bytes
} __attribute__((aligned(64))) attention_t; // Ensure the structure is aligned to 8 bytes

attention_t *Attention(int n_embd, int n_heads, int block_size);

#ifdef __cplusplus
}
#endif