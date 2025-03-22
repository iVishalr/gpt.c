#include <cuda_runtime.h>

static __device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
    }
    return x;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, offset, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, offset, 32);
    }
    return a;
}