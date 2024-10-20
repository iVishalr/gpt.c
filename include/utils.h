/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#pragma once

#include <stdio.h>
#include <stdbool.h>
#include "tensor.h"

#ifdef __cplusplus
extern "C"
{
#endif

// ----------------------------------------------------------------------------
// Generic error check macros

void check_error(bool condition, const char *file, int line, const char *msg, ...);

#define CHECK_ERROR(cond, msg, ...) check_error((cond), __FILE__, __LINE__, (msg), ##__VA_ARGS__)

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

FILE *fopen_check(const char *path, const char *mode, const char *file, int line);

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line);

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

void fclose_check(FILE *fp, const char *file, int line);

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

void fseek_check(FILE *fp, long off, int whence, const char *file, int line);

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

void *malloc_check(size_t size, const char *file, int line);

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

void *aligned_malloc_check(size_t alignment, size_t size, const char *file, int line);

#define alignedMallocCheck(alignment, size) aligned_malloc_check(alignment, size, __FILE__, __LINE__)

// ---------------------------------------------------------------------------------------------------
// print helpers

void print_table(char **key, char **value, size_t n);

// ---------------------------------------------------
// tensor checks

void tensor_check(const tensor_t *tensor, const char *file, int line);

#define tensorCheck(tensor) tensor_check(tensor, __FILE__, __LINE__)

void tensor_device_check(const tensor_t *A, const tensor_t *B, const char *file, int line);

#define tensorDeviceCheck(A, B) tensor_device_check(A, B, __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif