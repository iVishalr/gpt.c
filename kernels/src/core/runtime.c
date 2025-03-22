#include <core/runtime.h>
#include <cuda/runtime.h>


void runtime_init_dispatch(const device_t device) {
    if (device == CUDA)
        runtime_init_cuda();
}


void runtime_destroy_dispatch(const device_t device) {
    if (device == CUDA) 
        runtime_destroy_cuda();
}


void synchronize_dispatch(const device_t device) {
    if (device == CUDA)
        synchronize_cuda();
}