#ifndef STUBS_H
#define STUBS_H

#include "utils.h"  // Assumes CHECK_ERROR or a similar error-reporting mechanism is declared here

/**
 * DEFINE_CUDA_KERNEL_STUB - Macro to generate a stub implementation for a CUDA kernel function.
 *
 * @func:   The name of the function.
 * @params: The parameter list for the function (including types and parameter names).
 *
 * The generated function will immediately trigger an error indicating that CUDA support
 * is not available in this build.
 */
#define GENERATE_CUDA_KERNEL_STUB(func, params) \
    void func params { \
        CHECK_ERROR(1, "CUDA support is not available in this build (" #func ")."); \
    }

#endif // STUBS_H
