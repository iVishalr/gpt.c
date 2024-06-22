/**
 * \file            dataloader.h
 * \brief           Header file for dataloader
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

#include <stdio.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dataloader {
    // Integers grouped together
    int batch_size; // 4 bytes
    int block_size; // 4 bytes

    // Pointers grouped together
    FILE *fp;   // 8 bytes
    int *batch; // 8 bytes (assuming pointer size)

    // Size_t members (may vary in size)
    size_t _file_size;   // Size_t can vary (usually 8 bytes on 64-bit)
    size_t _curr_fp_ptr; // Size_t can vary (usually 8 bytes on 64-bit)

    // Function pointers grouped together
    void (*next)(struct dataloader *, tensor_t **); // 8 bytes
    void (*reset)(struct dataloader *);             // 8 bytes
    void (*free_layer)(struct dataloader *);        // 8 bytes
} __attribute__((aligned(8))) dataloader_t;         // Ensure the structure is aligned to 8 bytes

dataloader_t *DataLoader(const char *filename, const int batch_size, const int block_size);

#ifdef __cplusplus
}
#endif