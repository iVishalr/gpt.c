#include <stdio.h>
#include <stdlib.h>
#include "dataloader.h"

void dataloader_next(dataloader_t *loader, tensor_t **batch) {
    if (loader == NULL || batch == NULL)
        return;

    int batch_size, block_size;
    batch_size = loader->batch_size;
    block_size = loader->block_size;

    if (loader->batch == NULL)
        loader->batch = (int *)malloc(sizeof(int) * (batch_size * block_size + 1));
    
    if (loader->_curr_fp_ptr + (batch_size * block_size + 1) * sizeof(int) > loader->_file_size) {
        loader->_curr_fp_ptr = 0;
    }

    fseek(loader->fp, loader->_curr_fp_ptr, SEEK_SET);
    fread(loader->batch, sizeof(int), batch_size * block_size + 1, loader->fp);
    loader->_curr_fp_ptr = batch_size * block_size * sizeof(int);

    batch[0] = loader->batch;
    batch[1] = loader->batch + 1;
}

void dataloader_reset(dataloader_t *loader) {
    if (loader == NULL)
        return;

    loader->_curr_fp_ptr = 0;
}

void dataloader_free_layer(dataloader_t *loader) {
    if (loader == NULL)
        return;

    fclose(loader->fp);

    if (loader->batch)
        free(loader->batch);

    free(loader);
    return;
}

dataloader_t *DataLoader(const char *filename, const int batch_size, const int block_size) {
    if (filename == NULL)
        return NULL;

    dataloader_t *loader = (dataloader_t *)malloc(sizeof(dataloader_t));
    loader->batch_size = batch_size;
    loader->block_size = block_size;
    loader->batch = NULL;
    
    loader->fp = fopen(filename, "rb");
    if (loader->fp == NULL)  {
        printf("Error opening tokens file: %s.\n", filename);
        free(loader);
        exit(1);
    }
    loader->_curr_fp_ptr = 0;

    fseek(loader->fp, 0, SEEK_END);
    loader->_file_size = ftell(loader->fp);
    fseek(loader->fp, 0, SEEK_SET);

    if (loader->_file_size < (batch_size * block_size + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and block size given.\n");
        free(loader);
        exit(1);
    }

    loader->next = dataloader_next;
    loader->free_layer = dataloader_free_layer;
    loader->reset = dataloader_reset;
    return loader;
}