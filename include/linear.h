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

#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct linear {
    int in_features;
    int out_features;
    int use_bias;
    
    tensor_t *W;
    tensor_t *b;
    tensor_t *dW;
    tensor_t *db;
    tensor_t *cache;
    
    tensor_t *(*forward)(struct linear *, tensor_t *);
    tensor_t *(*backward)(struct linear *, tensor_t *);
    
    void (*description)(const struct linear *);
    int (*num_parameters)(const struct linear *);
    void (*free_layer)(struct linear *);
    
    tensor_t **(*parameters)(const struct linear *);
    tensor_t **(*gradients)(const struct linear *);
    
    int _num_param_tensors;
} linear_t;

linear_t *Linear(const int in_features, const int out_features, const int use_bias);

#ifdef __cplusplus
}
#endif