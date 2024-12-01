#pragma once

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define C10_WARP_SIZE 32

constexpr uint32_t num_threads() {
    return C10_WARP_SIZE * 4;
}

#if __CUDA_ARCH__ == 750
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1024;
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 1536;
#else
constexpr uint32_t CUDA_MAX_THREADS_PER_SM = 2048;
#endif
// CUDA_MAX_THREADS_PER_BLOCK is same for all architectures currently
constexpr uint32_t CUDA_MAX_THREADS_PER_BLOCK = 1024;
// CUDA_THREADS_PER_BLOCK_FALLBACK is the "canonical fallback" choice of block
// size. 256 is a good number for this fallback and should give good occupancy
// and versatility across all architectures.
constexpr uint32_t CUDA_THREADS_PER_BLOCK_FALLBACK = 256;
// NOTE: if you are thinking of constexpr-ify the inputs to launch bounds, it
//       turns out that although __launch_bounds__ can take constexpr, it
//       can't take a constexpr that has anything to do with templates.
//       Currently we use launch_bounds that depend on template arguments in
//       Loops.cuh, Reduce.cuh and LossCTC.cuh. Hence, C10_MAX_THREADS_PER_BLOCK
//       and C10_MIN_BLOCKS_PER_SM are kept as macros.
// Suppose you were planning to write __launch_bounds__(a, b), based on your
// performance tuning on a modern GPU. Instead, you should write
// __launch_bounds__(C10_MAX_THREADS_PER_BLOCK(a), C10_MIN_BLOCKS_PER_SM(a, b)),
// which will also properly respect limits on old architectures.
#define C10_MAX_THREADS_PER_BLOCK(val)             \
    (((val) <= CUDA_MAX_THREADS_PER_BLOCK) ? (val) \
                                           : CUDA_THREADS_PER_BLOCK_FALLBACK)
#define C10_MIN_BLOCKS_PER_SM(threads_per_block, blocks_per_sm)          \
    ((((threads_per_block) * (blocks_per_sm) <= CUDA_MAX_THREADS_PER_SM) \
          ? (blocks_per_sm)                                              \
          : ((CUDA_MAX_THREADS_PER_SM + (threads_per_block) - 1) /       \
             (threads_per_block))))
// C10_LAUNCH_BOUNDS is analogous to __launch_bounds__
#define C10_LAUNCH_BOUNDS_0 \
    __launch_bounds__(      \
        256, 4) // default launch bounds that should give good occupancy and
                // versatility across all architectures.
#define C10_LAUNCH_BOUNDS_1(max_threads_per_block) \
    __launch_bounds__((C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))))
#define C10_LAUNCH_BOUNDS_2(max_threads_per_block, min_blocks_per_sm) \
    __launch_bounds__(                                                \
        (C10_MAX_THREADS_PER_BLOCK((max_threads_per_block))),         \
        (C10_MIN_BLOCKS_PER_SM((max_threads_per_block), (min_blocks_per_sm))))

#ifdef __cplusplus
extern "C" {
#endif

inline void cuda_check(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}
#define cudaCheck(err) { cuda_check(err, __FILE__, __LINE__); }

inline void cublas_check(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("[cuBLAS ERROR]: %d %s %d\n", status, file, line);
        exit(EXIT_FAILURE);
    }
}
#define cublasCheck(status) { cublas_check((status), __FILE__, __LINE__); }

#ifdef __cplusplus
}
#endif