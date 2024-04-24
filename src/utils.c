#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

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

float rand_norm(float mu, float std) {
    return mu + gaussRandom() * std;
}

/* https : // stackoverflow.com/questions/11641629/generating-a-uniform-distribution-of-integers-in-c
*/
float rand_uniform(float low, float high)
{
    float r = rand() / (1.0f + RAND_MAX);
    float range = high - low + 1;
    float scaled = (r * range) + low;
    return scaled;
}

tensor_t *create(const int *shape, const int n) {
    if (shape == NULL) {
        printf("Expected required argument shape to be of type int ptr, but got NULL.\n");
        return NULL;
    }

    int length = n;
    int total_elements = 1;

    for (int i = 0; i < n; i++)
        total_elements *= shape[i];

    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));

    if (tensor == NULL) {
        printf("Could not allocate memory when creating tensor object.\n");
        exit(EXIT_FAILURE);
    }

    tensor->t = (float *)malloc(sizeof(float) * total_elements);

    if (tensor->t == NULL) {
        printf("Could not allocate memory when creating tensor.t object.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++)
        tensor->shape[i] = shape[i];

    tensor->ndims = n;
    tensor->length = total_elements;
    return tensor;
}

tensor_t *create_calloc(const int *shape, const int n)
{
    if (shape == NULL)
    {
        printf("Expected required argument shape to be of type int ptr, but got NULL.\n");
        return NULL;
    }

    int length = n;
    int total_elements = 1;

    for (int i = 0; i < n; i++)
        total_elements *= shape[i];

    tensor_t *tensor = (tensor_t *)malloc(sizeof(tensor_t));

    if (tensor == NULL)
    {
        printf("Could not allocate memory when creating tensor object.\n");
        exit(EXIT_FAILURE);
    }

    tensor->t = (float *)calloc(total_elements, sizeof(float));

    if (tensor->t == NULL)
    {
        printf("Could not allocate memory when creating tensor.t object.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++)
        tensor->shape[i] = shape[i];

    tensor->ndims = n;
    tensor->length = total_elements;
    return tensor;
}

tensor_t *create_tensor(const int *shape, const int n) {
    return create(shape, n);
}

tensor_t *randn(const int *shape, const int n) {
    tensor_t *tensor = create(shape, n);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        return NULL;
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = rand_norm(0.0f, 1.0f);

    return tensor;
}

tensor_t *zeros(const int *shape, const int n) {
    tensor_t *tensor = create_calloc(shape, n);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        return NULL;
    }

    return tensor;
}

tensor_t *ones(const int *shape, const int n) {
    tensor_t *tensor = create(shape, n);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        return NULL;
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = 1.0f;

    return tensor;
}

tensor_t *fill(const int *shape, const int n, const float value) {
    tensor_t *tensor = create(shape, n);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        return NULL;
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = value;

    return tensor;
}

tensor_t *empty(const int *shape, const int n) {
    tensor_t *tensor = create(shape, n);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        return NULL;
    }

    free(tensor->t);
    tensor->t = NULL;
    return tensor;
}

void *transpose(
    const int CORDER, const int CTRANS,
    const int crows, const int ccols,
    const float calpha, const tensor_t *A, const int clda,
    tensor_t *B, const int cldb) {
    
    if (A == NULL){
        printf("Expected required argument *A to be of type tensor_t, but got NULL.");
        return NULL;
    }

    if (B == NULL) {
        printf("Expected required argument *B to be of type tensor_t, but got NULL.");
        return NULL;
    }

    if (A->t == NULL || B->t == NULL) {
        printf("Expected tensor's underlying memory to be allocated when tensor was created, but found NULL.\n");
        return NULL;
    }

    cblas_somatcopy(CORDER, CTRANS, crows, ccols, calpha, A->t, clda, B->t, cldb);
    return B;
}

void *matmul(
    int Order,
    int TransA,
    int TransB,
    int M, int N, int K,
    const float alpha, const tensor_t *A, const int lda, const tensor_t *B, const int ldb, const float beta, tensor_t *C, const int ldc)
{
    if (A == NULL) {
        printf("Expected required argument *A to be of type tensor_t, but got NULL.");
        return NULL;
    }

    if (B == NULL) {
        printf("Expected required argument *B to be of type tensor_t, but got NULL.");
        return NULL;
    }

    if (C == NULL) {
        printf("Expected required argument *C to be of type tensor_t, but got NULL.");
        return NULL;
    }

    if (A->t == NULL || B->t == NULL || C->t == NULL) {
        printf("Expected tensor's underlying memory to be allocated when tensor was created, but found NULL.\n");
        return NULL;
    }

    cblas_sgemm(
        Order, TransA, TransB,
        M, N, K, 
        alpha, A->t, lda,
        B->t, ldb,
        beta, C->t, ldc
    );
    return C;
}

void *tensor_copy(tensor_t *dest, const tensor_t *src) {
    if (src == NULL || dest == NULL) {
        printf("Either src or dest ptr is NULL.\n");
        return NULL;
    }

    memcpy(dest->t, src->t, src->length * sizeof(float));
    
    for (int i = 0; i < src->ndims; i++)
        dest->shape[i] = src->shape[i];

    dest->ndims = src->ndims;
    dest->length = src->length;
    return dest;
}

void *uniform(tensor_t *tensor, const float low, const float hight) {
    if (tensor == NULL) {
        printf("Expected required argument *t to be of type tensor_t, but got NULL.");
        return NULL;
    }

    for (int i = 0; i < tensor->length; i++) {
        tensor->t[i] = rand_uniform(low, hight);
    }
    return tensor;
}

void free_tensor(tensor_t *tensor) {
    if (tensor == NULL)
        return;

    if (tensor->t != NULL)
        free(tensor->t);
    free(tensor);
}

void *shape(const tensor_t *tensor, char *shape) {
    if (tensor == NULL || shape == NULL) {
        return NULL;
    }

    int counter = 0;
    for (int i = 0; i < tensor->ndims; i++) {
        counter += sprintf(&shape[counter], "%d", tensor->shape[i]);
        counter += sprintf(&shape[counter], "%s", ", ");
    }
    shape[counter - 2] = '\0';
}

void *_shape(const tensor_t *tensor, char *_shape) {
    return shape(tensor, _shape);
}

void view(tensor_t *tensor, const int *shape, const int n) {
    if (tensor == NULL) {
        printf("Expected required argument *tensor to be of type tensor_t, but got NULL.\n");
        return;
    }

    int total_elements = 1;
    for (int i = 0; i < n; i++)
        total_elements *= shape[i];

    if (total_elements != tensor->length) {
        printf("Given shape is invalid for a tensor of size %d.\n", tensor->length);
        return;
    }

    for (int i = 0; i < n; i++)
        tensor->shape[i] = shape[i];

    tensor->ndims = n;
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
    print_tensor_helper(tensor, 0, 0, compact);
}
