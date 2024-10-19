#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "utils.h"

FILE *fopen_check(const char *path, const char *mode, const char *file, int line)
{
    FILE *fp = fopen(path, mode);
    if (fp == NULL)
    {
        fprintf(stderr, "Error: Failed to open file '%s' at %s:%d\n", path, file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Path: %s\n", path);
        fprintf(stderr, "  Mode: %s\n", mode);
        exit(EXIT_FAILURE);
    }
    return fp;
}

void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line)
{
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb)
    {
        if (feof(stream))
        {
            fprintf(stderr, "Error: Unexpected end of file at %s:%d\n", file, line);
        }
        else if (ferror(stream))
        {
            fprintf(stderr, "Error: File read error at %s:%d\n", file, line);
        }
        else
        {
            fprintf(stderr, "Error: Partial read at %s:%d. Expected %zu elements, read %zu\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Expected elements: %zu\n", nmemb);
        fprintf(stderr, "  Read elements: %zu\n", result);
        exit(EXIT_FAILURE);
    }
}

void fclose_check(FILE *fp, const char *file, int line)
{
    if (fclose(fp) != 0)
    {
        fprintf(stderr, "Error: Failed to close file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        exit(EXIT_FAILURE);
    }
}

void fseek_check(FILE *fp, long off, int whence, const char *file, int line)
{
    if (fseek(fp, off, whence) != 0)
    {
        fprintf(stderr, "Error: Failed to seek in file at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  Offset: %ld\n", off);
        fprintf(stderr, "  Whence: %d\n", whence);
        fprintf(stderr, "  File:   %s\n", file);
        fprintf(stderr, "  Line:   %d\n", line);
        exit(EXIT_FAILURE);
    }
}

void *malloc_check(size_t size, const char *file, int line)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *aligned_malloc_check(size_t alignment, size_t size, const char *file, int line)
{
    void *ptr = aligned_alloc(alignment, size);
    if (ptr == NULL)
    {
        fprintf(stderr, "Error: Aligned Memory allocation failed at %s:%d\n", file, line);
        fprintf(stderr, "Error details:\n");
        fprintf(stderr, "  File: %s\n", file);
        fprintf(stderr, "  Line: %d\n", line);
        fprintf(stderr, "  Size: %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void print_table(char **key, char **value, size_t n) {
    if (key == NULL || value == NULL || n <= 0)
        return;

    int max_key_width = 0, max_value_width = 0;
    for (int i = 0; i < n; i++) {
        if (key[i] == NULL || value[i] == NULL) continue;
        int k_len = strlen(key[i]);
        int v_len = strlen(value[i]);
        if (k_len > max_key_width) max_key_width = k_len;
        if (v_len > max_value_width) max_value_width = v_len;
    }
    char key_row[1024], val_row[1024];
    int _i = 0;
    for (_i = 0; _i < max_key_width; _i++)
        key_row[_i] = '-';
    key_row[_i] = '\0';
    for (_i = 0; _i < max_value_width; _i++)
        val_row[_i] = '-';
    val_row[_i] = '\0';

    printf("+-%-*s-+-%-*s-+\n", max_key_width, key_row, max_value_width, val_row);
    printf("| %-*s | %-*s |\n", max_key_width, "Parameter", max_value_width, "Value");
    printf("+-%-*s-+-%-*s-+\n", max_key_width, key_row, max_value_width, val_row);
    for (int i = 0; i < n; i++)
        printf("| %-*s | %-*s |\n", max_key_width, key[i], max_value_width, value[i]);
    printf("+-%-*s-+-%-*s-+\n", max_key_width, key_row, max_value_width, val_row);
}

void tensor_check(const tensor_t *tensor, const char *file, int line) {
    if (tensor == NULL) {
        printf("%s:%d Expected a pointer of type tensor_t, but got NULL\n", file, line);
        exit(EXIT_FAILURE);
    }

    int length = 1;
    for (int i = 0; i < tensor->ndims; i++)
        length *= tensor->shape[i];
    
    char tensor_shape[1024];
    shape(tensor, tensor_shape);

    if (length != tensor->length) {
        printf("%s:%d Expected a tensor of shape %s to be of length %d, but got a tensor of length %d\n", file, line, tensor_shape, length, tensor->length);
        exit(EXIT_FAILURE);
    }

    if (tensor->t == NULL) {
        printf("%s:%d Expected tensor->t to be a valid float pointer, but got NULL\n", file, line);
        exit(EXIT_FAILURE);
    }
}

void tensor_device_check(const tensor_t *A, const tensor_t *B, const char *file, int line) {
    if (A == NULL) {
        printf("%s:%d Expected A to be a pointer of type tensor_t, but got NULL\n", file, line);
        exit(EXIT_FAILURE);
    }

    if (B == NULL) {
        printf("%s:%d Expected B to be a pointer of type tensor_t, but got NULL\n", file, line);
        exit(EXIT_FAILURE);
    }


    char A_device[1024], B_device[1024];
    get_tensor_device(A, A_device);
    get_tensor_device(B, B_device);

    if (A->device != B->device) {
        printf("%s:%d Expected all tensors to be on same device, but found tensors present on %s and %s\n", file, line, A_device, B_device);
        exit(EXIT_FAILURE);
    }
}