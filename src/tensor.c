#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
#include "dispatch.h"
#include "tensor.h"
#include "utils.h"

float cache;
int return_cache;

float gaussGenerator(float *cache, int *return_cache) {
    if (*return_cache) {
        *return_cache = 0;
        return *cache;
    }
    // use drand48 to generate random values from uniform distribution
    float u = 2.0 * drand48() - 1.0;
    float v = 2.0 * drand48() - 1.0;
    float r = u * u + v * v;
    if (r == 0.0 || r > 1)
        return gaussGenerator(cache, return_cache);
    float c = sqrt(-2 * log(r) / r);
    *cache = c * v; // store this in cache
    *return_cache = 1;
    return u * c;
}

float gaussRandom() {
    cache = 0.0;
    return_cache = 0;
    return gaussGenerator(&cache, &return_cache);
}

// Implementation taken from
// https://stackoverflow.com/questions/11641629/generating-a-uniform-distribution-of-integers-in-c
float rand_uniform(float low, float high) {
    float r = rand() / (1.0f + (float)RAND_MAX);
    float range = high - low + 1;
    float scaled = (r * range) + low;
    return scaled;
}

// Box-Muller transform to generate a random number from a normal distribution
float rand_norm(double mean, double stddev) {
    static int have_spare = 0;
    static double spare;

    if (have_spare) {
        have_spare = 0;
        return mean + stddev * spare;
    }

    have_spare = 1;
    double u, v, s;
    do {
        u = rand_uniform(0, 1) * 2.0 - 1.0;
        v = rand_uniform(0, 1) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return (float)(mean + stddev * (u * s));
}

tensor_t *create_tensor(const int *shape, const int n, const device_t device) {
    CHECK_ERROR(shape == NULL, "Expected *shape to be a integer pointer, but got NULL.");

    tensor_t *tensor = (tensor_t *)mallocCheck(sizeof(tensor_t));

    tensor->ndims = n;
    tensor->device = device;
    tensor->to = move_tensor_data_dispatch;

    int total_elements = 1;
    for (int i = 0; i < n; i++) {
        total_elements *= shape[i];
        tensor->shape[i] = shape[i];
    }
    tensor->t = NULL;
    tensor->length = total_elements;
    create_tensor_data_dispatch(tensor);
    return tensor;
}


tensor_t *randn(const int *shape, const int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = rand_norm(0.0f, 1.0f);

    return tensor;
}


tensor_t *zeros(const int *shape, const int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    zeros_tensor_data_dispatch(tensor);
    return tensor;
}


tensor_t *ones(const int *shape, const int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    ones_tensor_data_dispatch(tensor);
    return tensor;
}


tensor_t *fill(const int *shape, const int n, const float value, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    fill_tensor_data_dispatch(tensor, value);
    return tensor;
}


tensor_t *arange(const int start, const int end, const int steps, const device_t device) {
    int num_elements = ceil((end - start) / steps);
    int shape[2] = {1, num_elements};
    tensor_t *tensor = create_tensor(shape, 2, device);
    arange_tensor_data_dispatch(tensor, start, end, steps);
    return tensor;
}


void tensor_copy(tensor_t *dest, const tensor_t *src) {

    CHECK_ERROR(src == NULL, "Expected *src to be a tensor_t pointer, but got NULL.");
    CHECK_ERROR(dest == NULL, "Expected *dest to be a tensor_t pointer, but got NULL.");

    copy_tensor_data_dispatch(dest, src);
    
    for (int i = 0; i < src->ndims; i++)
        dest->shape[i] = src->shape[i];

    dest->ndims = src->ndims;
    dest->length = src->length;
}


void saxpy(const int n, const float alpha, const tensor_t *x, const int offsetx, const int incx, tensor_t *y, const int offsety, const int incy) {
    CHECK_ERROR(x == NULL, "Expected *x to be a tensor_t pointer, but got NULL.");
    CHECK_ERROR(y == NULL, "Expected *y to be a tensor_t pointer, but got NULL.");

    saxpy_dispatch(n, alpha, x, offsetx, incx, y, offsety, incy);
}

void sgemm(
    const int TransA, const int TransB, const int M, const int N, const int K,
    const float alpha, const tensor_t *A, const int offsetA, const int lda,
    const tensor_t *B, const int offsetB, const int ldb,
    const float beta, tensor_t *C, const int offsetC, const int ldc
) {
    CHECK_ERROR(A == NULL, "Expected *A to be a tensor_t pointer, but got NULL.");
    CHECK_ERROR(B == NULL, "Expected *B to be a tensor_t pointer, but got NULL.");
    CHECK_ERROR(C == NULL, "Expected *C to be a tensor_t pointer, but got NULL.");

    sgemm_dispatch(TransA, TransB, M, N, K, alpha, A, offsetA, lda, B, offsetB, ldb, beta, C, offsetC, ldc);
}


void uniform(tensor_t *tensor, const float low, const float high) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer, but got NULL.");

    for (int i = 0; i < tensor->length; i++) {
        tensor->t[i] = rand_uniform(low, high);
    }
}

tensor_t *tensor_load(FILE *fp, const int *shape, int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    freadCheck(tensor->t, sizeof(float), tensor->length, fp);
    return tensor;
}

void tensor_save(FILE *fp, const tensor_t *tensor) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer, but got NULL.");
    fwrite(tensor->t, sizeof(float), tensor->length, fp);
}

void free_tensor(tensor_t *tensor) {
    if (tensor == NULL)
        return;

    if (tensor->t != NULL)
        free_dispatch(tensor->t, tensor->device);
    free(tensor);
}

void shape(const tensor_t *tensor, char *shape) {
    if (tensor == NULL || shape == NULL) {
        return;
    }

    int counter = 0;
    for (int i = 0; i < tensor->ndims; i++) {
        counter += sprintf(&shape[counter], "%d", tensor->shape[i]);
        counter += sprintf(&shape[counter], "%s", ", ");
    }
    shape[counter - 2] = '\0';
}

void view(tensor_t *tensor, const int *shape, const int n) {
    CHECK_ERROR(tensor == NULL, "Expected *tensor to be a tensor_t pointer, but got NULL.");

    int total_elements = 1;
    for (int i = 0; i < n; i++)
        total_elements *= shape[i];

    CHECK_ERROR(total_elements != tensor->length, "Given shape is invalid for a tensor of size %d.", tensor->length);

    for (int i = 0; i < n; i++)
        tensor->shape[i] = shape[i];

    tensor->ndims = n;
}

void get_tensor_device(const tensor_t *tensor, char *device) {
    CHECK_ERROR(device == NULL, "Expected *device to be a valid char pointer, but got NULL.");
    switch (tensor->device) {
        case CPU:
            sprintf(device, "%s", "cpu");
            break;
        case CUDA:
            sprintf(device, "%s", "cuda");
            break;
        default:
            printf("Tensor has an invalid device type\n");
            exit(EXIT_FAILURE);
            break;
    }
}

void print_shape(const tensor_t *tensor) {
    if (tensor == NULL) 
        return;

    char t_shape[1024];
    shape(tensor, t_shape);
    printf("(%s)\n", t_shape);
}

void print_tensor_helper(const tensor_t *tensor, const int dim, const int seek, const int compact) {
    
    char spaces[1024];
    int index = 0;
    for (int i = 0; i < dim; i++)
    {
        spaces[index] = ' ';
        spaces[index + 1] = ' ';
        index += 2;
    }
    spaces[index] = '\0';
    
    if (dim == tensor->ndims - 1) {
        printf("%s[ ", spaces);
        for (int i = 0; i < tensor->shape[dim]; i++) {
            if (compact) {
                if (i == tensor->shape[dim] - 1)
                    printf("%f ", tensor->t[seek * tensor->shape[dim] + i]);
                else if (i < 3 || i >= tensor->shape[dim] - 3)
                    printf("%f, ", tensor->t[seek * tensor->shape[dim] + i]);
                else if (i == 3)
                    printf("..., ");
                else if (i > 3 && i < tensor->shape[dim] - 3)
                    continue;
            } else {
                if (i == tensor->shape[dim] - 1)
                    printf("%f ", tensor->t[seek * tensor->shape[dim] + i]);
                else
                    printf("%f, ", tensor->t[seek * tensor->shape[dim] + i]);
            }
        }
        if (tensor->ndims == 1)
            printf("]\n");
        else
            printf("],\n");
        return;
    }

    printf("%s[\n", spaces);
    for (int i = 0; i < tensor->shape[dim]; i++)
        print_tensor_helper(tensor, dim + 1, seek * tensor->shape[dim] + i, compact);
    
    if (dim > 0)
        printf("%s],\n", spaces);
    else
        printf("%s]\n", spaces);
    
    return;
}

void print_tensor(const tensor_t *tensor, const int compact) {
    if (tensor == NULL) {
        printf("NULL\n");
        return;
    }
    print_tensor_helper(tensor, 0, 0, compact);
}
