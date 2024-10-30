#include <cblas.h>
#include <cpu/Alloc.h>
#include <cpu/Tensor.h>
#include <common/kutils.h>

void move_tensor_to_host_cpu(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(tensor->t == NULL, "Expected *tensor->t to be a float pointer. Got NULL");
    // noop as CPU kernel is already the host device.
}

void move_tensor_to_device_cpu(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(tensor->t == NULL, "Expected *tensor->t to be a float pointer. Got NULL");
    // noop as CPU kernel is already the host device.
}

void create_tensor_data_cpu(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    tensor->t = (float*)AllocCheck(aligned_alloc_cpu, tensor->length * sizeof(float), 64);
}

void zeros_tensor_data_cpu(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    if (tensor->t == NULL) create_tensor_data_cpu(tensor);
    
    float *data = __builtin_assume_aligned(tensor->t, 64);
    for (int i = 0; i < tensor->length; i++)
        data[i] = 0.0f;
}

void ones_tensor_data_cpu(tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    if (tensor->t == NULL) create_tensor_data_cpu(tensor);
    
    float *data = __builtin_assume_aligned(tensor->t, 64);
    for (int i = 0; i < tensor->length; i++)
        data[i] = 1.0f;
}

void fill_tensor_data_cpu(tensor_t *tensor, const float value) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    if (tensor->t == NULL) create_tensor_data_cpu(tensor);
    
    float *data = __builtin_assume_aligned(tensor->t, 64);
    for (int i = 0; i < tensor->length; i++)
        data[i] = value;
}

void copy_tensor_data_cpu(tensor_t *dst, const tensor_t *src) {
    CHECK_ERROR(dst == NULL, "Expected *dst to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(src == NULL, "Expected *src to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(dst->t == NULL, "Expected *dst->t to be a float pointer. Got NULL");
    CHECK_ERROR(src->t == NULL, "Expected *src->t to be a float pointer. Got NULL");

    const float *_src = __builtin_assume_aligned(src->t, 64);
    float *_dst = __builtin_assume_aligned(dst->t, 64);
    cblas_scopy(src->length, _src, 1, _dst, 1);
}

void saxpy_cpu(
    const int n, const float alpha,
    const tensor_t *x, const int offsetx, const int incx,
    tensor_t *y, const int offsety, const int incy
) {
    const float *_x = __builtin_assume_aligned(x->t, 64);
    float *_y = __builtin_assume_aligned(y->t, 64);
    cblas_saxpy(n, alpha, _x + offsetx, incx, _y + offsety, incy);
}