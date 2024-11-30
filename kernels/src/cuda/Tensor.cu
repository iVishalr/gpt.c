#include <cpu/Alloc.h>
#include <cuda/Alloc.h>
#include <cuda/Tensor.h>
#include <cuda/cuda_common.h>

#include <common/kutils.h>

#ifdef __cplusplus
extern "C" {
#endif

void move_tensor_to_host_cuda(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(tensor->t == NULL, "Expected *tensor->t to be a float pointer. Got NULL");

    // if tensor is already present on host, return
    if (tensor->device == CPU) return;

    float *device_ptr = tensor->t;
    float *host_ptr = (float*)AllocCheck(aligned_alloc_cpu, tensor->length * sizeof(float), 64);
    cudaCheck(cudaMemcpy(host_ptr, device_ptr, tensor->length * sizeof(float), cudaMemcpyDeviceToHost));

    tensor->t = host_ptr;
    tensor->device = CPU;
    free_cuda(device_ptr);
}

void move_tensor_to_device_cuda(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(tensor->t == NULL, "Expected *tensor->t to be a float pointer. Got NULL");

    // if tensor is already present on device, return
    if (tensor->device == CUDA) return;

    float *host_ptr = tensor->t;
    float *device_ptr;
    const size_t size = tensor->length * sizeof(float);
    cudaCheck(cudaMalloc((void**)&device_ptr, size));
    cudaCheck(cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice));

    tensor->t = device_ptr;
    tensor->device = CUDA;
    free_cpu(host_ptr);
}

#ifdef __cplusplus
}
#endif