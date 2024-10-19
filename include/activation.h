/**
 * \file            gelu.h
 * \brief           Header file for GELU layer
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

typedef struct gelu {
    tensor_t *cache;                                  // 8 bytes
    tensor_t *(*forward)(struct gelu *, tensor_t *);  // 8 bytes
    tensor_t *(*backward)(struct gelu *, tensor_t *); // 8 bytes
    void (*description)(const struct gelu *);         // 8 bytes
    void (*free_layer)(struct gelu *);                // 8 bytes
    void (*free_cache)(struct gelu *);                // 8 bytes
    void (*to)(struct gelu *, const device_t device); // 8 bytes
} __attribute__((aligned(64))) gelu_t;

typedef struct softmax {
    tensor_t *cache;                                     // 8 bytes
    tensor_t *(*forward)(struct softmax *, tensor_t *);  // 8 bytes
    tensor_t *(*backward)(struct softmax *, tensor_t *); // 8 bytes
    void (*description)(const struct softmax *);         // 8 bytes
    void (*free_layer)(struct softmax *);                // 8 bytes
    void (*free_cache)(struct softmax *);                // 8 bytes
    void (*to)(struct softmax *, const device_t device); // 8 bytes
} __attribute__((aligned(64))) softmax_t;

gelu_t *GELU();
softmax_t *Softmax();


#ifdef __cplusplus
}
#endif