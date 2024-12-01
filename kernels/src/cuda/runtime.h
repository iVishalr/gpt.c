#pragma once

#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

extern cublasHandle_t cublas_handle;

void setup_cublas_handle();
cublasHandle_t get_cublas_handle();

#ifdef __cplusplus
}
#endif