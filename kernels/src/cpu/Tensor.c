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

void arange_tensor_data_cpu(tensor_t *tensor, const int start, const int end, const int steps) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer. Got NULL");
    if (tensor->t == NULL) create_tensor_data_cpu(tensor);

    float *data = __builtin_assume_aligned(tensor->t, 64);
    for (int i = 0; i < tensor->length; i++)
        data[i] = start + steps * i;
}

void copy_tensor_data_cpu(tensor_t *dst, const tensor_t *src) {
    CHECK_ERROR(dst == NULL, "Expected *dst to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(src == NULL, "Expected *src to be a tensor_t pointer. Got NULL");
    CHECK_ERROR(dst->t == NULL, "Expected *dst->t to be a float pointer. Got NULL");
    CHECK_ERROR(src->t == NULL, "Expected *src->t to be a float pointer. Got NULL");
    CHECK_ERROR(src->length != dst->length, "Expected src and dst tensors to be of same length. Got %d != %d", src->length, dst->length);

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

void sgemm_cpu(
    const int TransA, const int TransB, const int M, const int N, const int K,
    const float alpha, const tensor_t *A, const int offsetA, const int lda,
    const tensor_t *B, const int offsetB, const int ldb,
    const float beta, tensor_t *C, const int offsetC, const int ldc
) {
    const float *_A = __builtin_assume_aligned(A->t, 64);
    const float *_B = __builtin_assume_aligned(B->t, 64);
    float *_C = __builtin_assume_aligned(C->t, 64);

    enum CBLAS_TRANSPOSE transA = CblasNoTrans, transB = CblasNoTrans;

    if (TransA == 1)
        transA = CblasTrans;
    if (TransB == 1)
        transB = CblasTrans;

    cblas_sgemm(
        CblasRowMajor, transA, transB, M, N, K,
        alpha, _A + offsetA, lda,
        _B + offsetB, ldb,
        beta, _C + offsetC, ldc
    );
}