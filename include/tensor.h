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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensor {
    float *t;
    int ndims;
    int length;
    int shape[1024];
} tensor_t;

tensor_t *create_tensor(const int *shape, const int n);
tensor_t *randn(const int *shape, const int n);
tensor_t *zeros(const int *shape, const int n);
tensor_t *ones(const int *shape, const int n);
tensor_t *fill(const int *shape, const int n, const float value);
tensor_t *empty(const int *shape, const int n);

void *transpose(
    const int CORDER, const int CTRANS,
    const int crows, const int ccols,
    const float calpha, const tensor_t *A, const int clda,
    tensor_t *B, const int cldb
);

void *matmul(
    int Order,
    int TransA,
    int TransB,
    int M, int N, int K,
    const float alpha, const tensor_t *A, const int lda, const tensor_t *B, const int ldb, const float beta, tensor_t *C, const int ldc
);

void mul_(tensor_t *x, const float s);
void pow_(tensor_t *x, const float p);
void *tensor_copy(tensor_t *dest, const tensor_t *src);
void *uniform(tensor_t *tensor, const float low, const float high);
void *shape(const tensor_t *tensor, char *shape);
void view(tensor_t *tensor, const int *shape, const int n);

tensor_t *tensor_load(FILE *fp, const int *shape, int n);
void free_tensor(tensor_t *tensor);
void print_tensor(const tensor_t *tensor, const int compact);
void print_shape(const tensor_t *tensor);

#ifdef __cplusplus
}
#endif