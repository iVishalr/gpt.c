/**
 * \file            tokenizer.h
 * \brief           Header file for tokenizer
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

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tokenizer {
    uint32_t vocab_size;    // 4 bytes
    int eot_token;          // 4 bytes
    char **token_map;       // 8 bytes

    // Function pointers grouped together
    void (*free_layer)(struct tokenizer *);                                                                 // 8 bytes
    char *(*decode)(struct tokenizer *, uint32_t token);                                                    // 8 bytes   
    void (*decode_tokens)(struct tokenizer *, uint32_t *tokens, size_t n, char *dest, size_t dest_n);      // 8 bytes
    uint32_t (*decode_length)(struct tokenizer *, uint32_t *tokens, size_t n);                              // 8 bytes
} tokenizer_t;

void safe_printf(const char *piece);
tokenizer_t *Tokenizer(const char *filename);

#ifdef __cplusplus
}
#endif