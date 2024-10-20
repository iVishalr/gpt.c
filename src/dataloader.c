#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "dataloader.h"


void dataloader_next(dataloader_t *loader, tensor_t **batch);
int dataloader_len(dataloader_t *loader);
void dataloader_reset(dataloader_t *loader);
void dataloader_free_layer(dataloader_t *loader);


// DataLoader Class
dataloader_t *DataLoader(const char *filename, const int batch_size, const int block_size) {
    if (filename == NULL)
        exit(EXIT_FAILURE);

    dataloader_t *loader = (dataloader_t *)mallocCheck(sizeof(dataloader_t));
    loader->batch_size = batch_size;
    loader->block_size = block_size;
    loader->batch = NULL;

    loader->fp = fopenCheck(filename, "rb");
    if (loader->fp == NULL) {
        printf("Error opening tokens file: %s.\n", filename);
        free(loader);
        exit(EXIT_FAILURE);
    }
    loader->_curr_fp_ptr = 0;

    fseek(loader->fp, 0, SEEK_END);
    loader->_file_size = ftell(loader->fp);
    fseek(loader->fp, 0, SEEK_SET);

    if (loader->_file_size < (batch_size * block_size + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and block size given.\n");
        free(loader);
        exit(EXIT_FAILURE);
    }

    loader->next = dataloader_next;
    loader->len = dataloader_len;
    loader->free_layer = dataloader_free_layer;
    loader->reset = dataloader_reset;
    return loader;
}


void dataloader_next(dataloader_t *loader, tensor_t **batch) {
    if (loader == NULL || batch == NULL)
        exit(EXIT_FAILURE);

    int batch_size, block_size;
    batch_size = loader->batch_size;
    block_size = loader->block_size;

    if (loader->batch == NULL)
        loader->batch = (int *)mallocCheck(sizeof(int) * (batch_size * block_size + 1));
    
    if (loader->_curr_fp_ptr + (batch_size * block_size + 1) * sizeof(int) > loader->_file_size) {
        loader->_curr_fp_ptr = 0;
    }

    fseekCheck(loader->fp, loader->_curr_fp_ptr, SEEK_SET);
    freadCheck(loader->batch, sizeof(int), batch_size * block_size + 1, loader->fp);
    loader->_curr_fp_ptr += batch_size * block_size * sizeof(int);

    int input_shape[2] = {batch_size, block_size};
    tensor_t *inputs = create_tensor(input_shape, 2, CPU);
    tensor_t *targets = create_tensor(input_shape, 2, CPU);

    for (int i = 0; i < batch_size * block_size; i++) {
        inputs->t[i] = (int)loader->batch[i];
        targets->t[i] = (int)loader->batch[i+1];
    }

    batch[0] = inputs;
    batch[1] = targets;
}


int dataloader_len(dataloader_t *loader) {
    if (loader == NULL)
        exit(EXIT_FAILURE);

    return (int)loader->_file_size / ((loader->batch_size * loader->block_size + 1) * sizeof(int));
}


void dataloader_reset(dataloader_t *loader) {
    if (loader == NULL)
        return;

    loader->_curr_fp_ptr = 0;
}


void dataloader_free_layer(dataloader_t *loader) {
    if (loader == NULL)
        return;

    if (loader->fp)
        fcloseCheck(loader->fp);

    if (loader->batch)
        free(loader->batch);

    free(loader);
    return;
}
