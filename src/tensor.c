#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>
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
float rand_uniform(float low, float high)
{
    float r = rand() / (1.0f + (float)RAND_MAX);
    float range = high - low + 1;
    float scaled = (r * range) + low;
    return scaled;
}

// Box-Muller transform to generate a random number from a normal distribution
float rand_norm(double mean, double stddev)
{
    static int have_spare = 0;
    static double spare;

    if (have_spare)
    {
        have_spare = 0;
        return mean + stddev * spare;
    }

    have_spare = 1;
    double u, v, s;
    do
    {
        u = rand_uniform(0, 1) * 2.0 - 1.0;
        v = rand_uniform(0, 1) * 2.0 - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    return (float)(mean + stddev * (u * s));
}

void __move_tensor_to_device(tensor_t *tensor, const device_t device) {}

tensor_t *create_tensor(const int *shape, const int n, const device_t device) {
    if (shape == NULL) {
        printf("Expected required argument shape to be of type int ptr, but got NULL.\n");
        return NULL;
    }


    tensor_t *tensor = (tensor_t *)mallocCheck(sizeof(tensor_t));

    tensor->ndims = n;
    tensor->device = device;
    tensor->to = __move_tensor_to_device;

    int total_elements = 1;
    for (int i = 0; i < n; i++) {
        total_elements *= shape[i];
        tensor->shape[i] = shape[i];
    }

    tensor->length = total_elements;
    tensor->t = (float *)alignedMallocCheck(64, total_elements * sizeof(float));
    return tensor;
}

tensor_t *randn(const int *shape, const int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = rand_norm(0.0f, 1.0f);

    return tensor;
}

tensor_t *zeros(const int *shape, const int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = 0.0f;

    return tensor;
}

tensor_t *ones(const int *shape, const int n, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = 1.0f;

    return tensor;
}

tensor_t *fill(const int *shape, const int n, const float value, const device_t device) {
    tensor_t *tensor = create_tensor(shape, n, device);
    if (tensor == NULL) {
        printf("Error when creating tensor object.\n");
        return NULL;
    }

    for (int i = 0; i < tensor->length; i++)
        tensor->t[i] = value;

    return tensor;
}

void mul_(tensor_t *x, const float s) {
    if (x == NULL) {
        printf("Required argument *x is NULL.\n");
        exit(EXIT_FAILURE);
    }
    
    cblas_sscal(x->length, s, x->t, 1);
}

void pow_(tensor_t *x, const float p) {
    if (x == NULL) {
        printf("Required argument *x is NULL.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < x->length; i++)
        x->t[i] = powf(x->t[i], p);
}

void tensor_copy(tensor_t *dest, const tensor_t *src) {
    if (src == NULL || dest == NULL) {
        printf("Either src or dest ptr is NULL.\n");
        exit(EXIT_FAILURE);
    }

    cblas_scopy(src->length, src->t, 1, dest->t, 1);
    
    for (int i = 0; i < src->ndims; i++)
        dest->shape[i] = src->shape[i];

    dest->ndims = src->ndims;
    dest->length = src->length;
}

void uniform(tensor_t *tensor, const float low, const float high) {
    if (tensor == NULL) {
        printf("Expected required argument *t to be of type tensor_t, but got NULL.");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < tensor->length; i++) {
        tensor->t[i] = rand_uniform(low, high);
    }
}

tensor_t *tensor_load(FILE *fp, const int *shape, int n, const device_t device) {
    if (fp == NULL) {
        printf("Invalid FILE ptr *fp.\n");
        exit(EXIT_FAILURE);
    }

    tensor_t *tensor = create_tensor(shape, n, device);
    freadCheck(tensor->t, sizeof(float), tensor->length, fp);
    return tensor;
}

void tensor_save(FILE *fp, const tensor_t *tensor) {
    if (fp == NULL) {
        printf("Invalid FILE ptr *fp.\n");
        exit(EXIT_FAILURE);
    }

    if (tensor == NULL) {
        printf("Expected *tensor to be of type tensor_t. Got NULL.\n");
        exit(EXIT_FAILURE);
    }

    fwrite(tensor->t, sizeof(float), tensor->length, fp);
}

void free_tensor(tensor_t *tensor) {
    if (tensor == NULL)
        return;

    if (tensor->t != NULL)
        free(tensor->t);
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
    if (tensor == NULL) {
        printf("Expected required argument *tensor to be of type tensor_t, but got NULL.\n");
        exit(EXIT_FAILURE);
    }

    int total_elements = 1;
    for (int i = 0; i < n; i++)
        total_elements *= shape[i];

    if (total_elements != tensor->length) {
        printf("Given shape is invalid for a tensor of size %d.\n", tensor->length);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++)
        tensor->shape[i] = shape[i];

    tensor->ndims = n;
}

void get_tensor_device(const tensor_t *tensor, char *device) {
    if (device == NULL) {
        printf("Expected *device to be a valid char pointer, but got NULL\n");
        exit(EXIT_FAILURE);
    }
    switch (tensor->device)
    {
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
