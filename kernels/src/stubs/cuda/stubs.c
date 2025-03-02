
#include <cuda/AdamW.h>
#include <cuda/Alloc.h>
#include <cuda/Attention.h>
#include <cuda/CrossEntropyLoss.h>
#include <cuda/Embedding.h>
#include <cuda/GeLU.h>
#include <cuda/LayerNorm.h>
#include <cuda/Linear.h>
#include <cuda/runtime.h>
#include <cuda/Softmax.h>
#include <cuda/Tensor.h>

#include <stubs/cuda/stubs.h>

// Stubs for AdamW.h
GENERATE_CUDA_KERNEL_STUB(
    step_adamW_cuda_kernel, (
        tensor_t **parameters,
        tensor_t **gradients,
        tensor_t **m,
        tensor_t **v,
        const int n,
        const float lr,
        const float beta1,
        const float beta2,
        const float weight_decay,
        const float eps,
        const int step
    )
);

GENERATE_CUDA_KERNEL_STUB(
    zero_grad_adamW_cuda_kernel, (
        tensor_t **gradients, 
        const int n
    )
);

// Stubs for Alloc.h
GENERATE_CUDA_KERNEL_STUB(
    *alloc_cuda, (
        const size_t nbytes
    )
);

GENERATE_CUDA_KERNEL_STUB(
    free_cuda, (
        void *ptr
    )
);

// Stubs for Attention.h
GENERATE_CUDA_KERNEL_STUB(
    attention_forward_cuda_kernel, (
        const tensor_t *input,
        const tensor_t *mask,
        const int n_heads,
        tensor_t **cache,
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    attention_backward_cuda_kernel, (
        const tensor_t *global_grad, 
        tensor_t **cache,
        const int n_heads,
        tensor_t *dout
    )
);

// Stubs for CrossEntropyLoss.h
GENERATE_CUDA_KERNEL_STUB(
    cross_entropy_forward_cuda_kernel, (
        const tensor_t *logits,
        const tensor_t *targets,
        tensor_t **cache,
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    cross_entropy_backward_cuda_kernel, (
        const tensor_t *global_grad,
        const tensor_t **cache,
        tensor_t *dout
    )
);

// Stubs for Embedding.h
GENERATE_CUDA_KERNEL_STUB(
    embedding_forward_cuda_kernel, (
        const tensor_t *W, 
        const tensor_t *input, 
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    embedding_backward_cuda_kernel, (
        const tensor_t *global_grad,
        const tensor_t *cache,
        tensor_t *dW
    )
);

// Stubs for GeLU.h
GENERATE_CUDA_KERNEL_STUB(
    gelu_forward_cuda_kernel, (
        const tensor_t *input, 
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    gelu_backward_cuda_kernel, (
        const tensor_t *global_grad,
        const tensor_t *cache,
        tensor_t *dout
    )
);

// Stubs for LayerNorm.h
GENERATE_CUDA_KERNEL_STUB(
    layer_norm_forward_cuda_kernel, (
        const tensor_t *W, 
        const tensor_t *b,
        const tensor_t *input,
        const float eps,
        tensor_t **cache,
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    layer_norm_backward_cuda_kernel, (
        const tensor_t *global_grad,
        const tensor_t **cache,
        const tensor_t *W,
        tensor_t *dW,
        tensor_t *db,
        tensor_t *dout
    )
);

// Stubs for Linear.h
GENERATE_CUDA_KERNEL_STUB(
    linear_forward_cuda_kernel, (
        const tensor_t *W,
        const tensor_t *b,
        const tensor_t *input,
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    linear_backward_cuda_kernel, (
        const tensor_t *global_grad,
        const tensor_t *cache,
        const tensor_t *W,
        tensor_t *dW,
        tensor_t *db,
        tensor_t *dout
    )
);

// Stubs for runtime.h
GENERATE_CUDA_KERNEL_STUB(
    runtime_init_cuda, ()
);

GENERATE_CUDA_KERNEL_STUB(
    runtime_destroy_cuda, ()
);

GENERATE_CUDA_KERNEL_STUB(
    synchronize_cuda, ()
);

// Stubs for Softmax.h
GENERATE_CUDA_KERNEL_STUB(
    softmax_forward_cuda_kernel, (
        const tensor_t *input,
        tensor_t *output
    )
);

GENERATE_CUDA_KERNEL_STUB(
    softmax_backward_cuda_kernel, (
        const tensor_t *global_grad,
        const tensor_t *cache,
        tensor_t *dout
    )
);

// Stubs for Tensor.h
GENERATE_CUDA_KERNEL_STUB(
    move_tensor_to_host_cuda, (tensor_t *tensor)
);

GENERATE_CUDA_KERNEL_STUB(
    move_tensor_to_device_cuda, (tensor_t *tensor)
)

GENERATE_CUDA_KERNEL_STUB(
    create_tensor_data_cuda, (tensor_t *tensor)
);

GENERATE_CUDA_KERNEL_STUB(
    zeros_tensor_data_cuda, (tensor_t *tensor)
);

GENERATE_CUDA_KERNEL_STUB(
    ones_tensor_data_cuda, (tensor_t *tensor)
);

GENERATE_CUDA_KERNEL_STUB(
    fill_tensor_data_cuda, (tensor_t *tensor, const float value)
);

GENERATE_CUDA_KERNEL_STUB(
    arange_tensor_data_cuda, (tensor_t *tensor, const int start, const int end, const int steps)
);

GENERATE_CUDA_KERNEL_STUB(
    copy_tensor_data_cuda, (tensor_t *dst, const tensor_t *src)
);

GENERATE_CUDA_KERNEL_STUB(
    saxpy_cuda, (
        const int n, const float alpha, 
        const tensor_t *x, const int offsetx, const int incx, 
        tensor_t *y, const int offsety, const int incy
    )
);

GENERATE_CUDA_KERNEL_STUB(
    sgemm_cuda, (
        const int TransA, const int TransB, const int M, const int N, const int K,
        const float alpha, const tensor_t *A, const int offsetA, const int lda,
        const tensor_t *B, const int offsetB, const int ldb, 
        const float beta, tensor_t *C, const int offsetC, const int ldc
    )
);

GENERATE_CUDA_KERNEL_STUB(
    sgemm_strided_batched_cuda, (
        const int TransA, const int TransB, const int M, const int N, const int K,
        const float alpha, const tensor_t *A, const int lda, const int strideA,
        const tensor_t *B, const int ldb, const int strideB,
        const float beta, tensor_t *C, const int ldc, const int strideC, const int batch_count
    )
);
