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
    CHECK_ERROR(filename == NULL, "Expected *filename to be a const char pointer, but got NULL.");
    CHECK_ERROR(batch_size <= 0, "Expected batch_size to be a positive integer, but got %d.", batch_size);
    CHECK_ERROR(block_size <= 0, "Expected block_size to be a positive integer, but got %d.", block_size);

    dataloader_t *loader = (dataloader_t *)mallocCheck(sizeof(dataloader_t));
    loader->batch_size = batch_size;
    loader->block_size = block_size;
    loader->batch = NULL;

    loader->fp = fopenCheck(filename, "rb");
    loader->_curr_fp_ptr = 0;

    fseekCheck(loader->fp, 0, SEEK_END);
    loader->_file_size = ftell(loader->fp);
    fseekCheck(loader->fp, 0, SEEK_SET);

    CHECK_ERROR(
        loader->_file_size < (batch_size * block_size + 1) * sizeof(int),
        "File size (%zu) is too small for the batch_size (%d) and block_size (%d) given.", loader->_file_size, batch_size, block_size
    );

    loader->next = dataloader_next;
    loader->len = dataloader_len;
    loader->free_layer = dataloader_free_layer;
    loader->reset = dataloader_reset;
    return loader;
}


void dataloader_next(dataloader_t *loader, tensor_t **batch) {
    CHECK_ERROR(loader == NULL, "Expected *loader to be a dataloader_t pointer, but got NULL.");
    CHECK_ERROR(batch == NULL, "Expected **batch to be a tensor_t pointer, but got NULL.");

    int batch_size, block_size;
    batch_size = loader->batch_size;
    block_size = loader->block_size;

    if (loader->batch == NULL)
        loader->batch = (int *)alignedMallocCheck(64, sizeof(int) * (batch_size * block_size + 1));
    
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
        inputs->t[i] = (float)loader->batch[i];
        targets->t[i] = (float)loader->batch[i+1];
    }

    batch[0] = inputs;
    batch[1] = targets;
}


int dataloader_len(dataloader_t *loader) {
    CHECK_ERROR(loader == NULL, "Expected *loader to be a dataloader_t pointer, but got NULL.");
    return (int)loader->_file_size / ((loader->batch_size * loader->block_size + 1) * sizeof(int));
}


void dataloader_reset(dataloader_t *loader) {
    CHECK_ERROR(loader == NULL, "Expected *loader to be a dataloader_t pointer, but got NULL.");
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
