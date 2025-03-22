#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void runtime_init_cuda();
void runtime_destroy_cuda();
void synchronize_cuda();

#ifdef __cplusplus
}
#endif