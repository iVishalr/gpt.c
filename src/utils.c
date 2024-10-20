#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "tensor.h"
#include "utils.h"

void check_error(bool condition, const char *file, int line, const char *msg, ...) {
    if (condition) {
        va_list args;
        va_start(args, msg);
        fprintf(stderr, "[ERROR] %s:%d: ", file, line);
        vfprintf(stderr, msg, args);
        fprintf(stderr, "\n");
        va_end(args);
        exit(EXIT_FAILURE);
    }
}


FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    check_error((fp == NULL), file, line, ("Failed to open file '%s'."), path);
    return fp;
}


void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {

        check_error(feof(stream), file, line, "Unexpected end of file. Expected to read %zu elements, but read %zu elements.", nmemb, result);
        check_error(ferror(stream), file, line, "Error reading file. Expected to read %zu elements, but read %zu elements.", nmemb, result);
        check_error(!(feof(stream) && ferror), file, line, "File was partially read. Expected to read %zu elements, but read %zu elements.", nmemb, result);
    }
}

void fclose_check(FILE *fp, const char *file, int line) {
    int status = fclose(fp);
    check_error(status != 0, file, line, "Failed to close file.");
}

void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
    int status = fseek(fp, off, whence);
    check_error(status != 0, file, line, "Failed to seek in file. Offset: %ld, Whence: %d", off, whence);
}

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    check_error(ptr == NULL, file, line, "Failed to allocate memory. Requested allocation size = %zu bytes.", size);
    return ptr;
}

void *aligned_malloc_check(size_t alignment, size_t size, const char *file, int line) {
    void *ptr = aligned_alloc(alignment, size);
    check_error(ptr == NULL, file, line, "Failed to allocate aligned memory. Requested allocation size = %zu bytes.", size);
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