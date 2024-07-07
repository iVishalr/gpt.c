#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "transformer.h"
#include "optim.h"
#include "dataloader.h"
#include "utils.h"

#define True 1
#define False 0

// main training loop
const char *tiny_shakespeare_train = "data/tiny_shakespeare/tiny_shakespeare_train.bin";
const char *tiny_shakespeare_val = "data/tiny_shakespeare/tiny_shakespeare_val.bin";
const char *checkpoint_path = "logs/checkpoint.bin";
const char *c_checkpoint_path = "logs/c_checkpoint.bin";

// training settings
const int batch_size = 8;
const int block_size = 128;
const int training_steps = 298;
const float lr = 3e-4f;
const float beta1 = 0.9f;
const float beta2 = 0.999f;
const float eps = 1e-8f;
const float weight_decay = 0.00f;

// validation settings
const int validation_batch_size = 8;
const int validation_block_size = 128;
const int validation_interval = 50;


int load_model(const char *file_path, gpt2_t *model) {
    if (model == NULL) {
        printf("Expected *model to be of type gpt2_t. Got NULL.\n");
        exit(1);
    }

    FILE *fp = fopenCheck(file_path, "rb");
    printf("Loading checkpoint from %s\n",file_path);
    int headers[256];
    freadCheck(headers, sizeof(int), 256, fp);
    if (headers[0] != 20240415) {
        printf("Bad magic model file\n");
        fcloseCheck(fp);
        exit(1);
    }
    size_t max_block_size, vocab_size, n_layers, n_heads, n_embd;
    size_t shape_header_size, steps;
    max_block_size = headers[1];
    vocab_size = headers[2];
    n_layers = headers[3];
    n_heads = headers[4];
    n_embd = headers[5];
    shape_header_size = headers[6];
    steps = headers[7];

    // validate model configurations
    int config_valid = 1;
    if (model->block_size != max_block_size) {
        printf("ValueError: model->block_size does not match block_size in checkpoint. Got %d != %zu\n", model->block_size, max_block_size);
        config_valid = 0;
    } 
    if (model->vocab_size != vocab_size) {
        printf("ValueError: model->vocab_size does not match vocab_size in checkpoint. Got %d != %zu\n", model->vocab_size, vocab_size);
        config_valid = 0;
    }
    if (model->n_layers != n_layers) {
        printf("ValueError: model->n_layers does not match n_layers in checkpoint. Got %d != %zu\n", model->n_layers, n_layers);
        config_valid = 0;
    }
    if (model->n_heads != n_heads) {
        printf("ValueError: model->n_heads does not match n_heads in checkpoint. Got %d != %zu\n", model->n_heads, n_heads);
        config_valid = 0;
    }
    if (model->n_embd != n_embd) {
        printf("ValueError: model->n_embd does not match n_embd in checkpoint. Got %d != %zu\n", model->n_embd, n_embd);
        config_valid = 0;
    }

    if (!config_valid) {
        fcloseCheck(fp);
        exit(1);
    }

    printf("[GPT2 | steps trained: %zu]\n", steps);
    printf("max_block_size: %zu\n", max_block_size);
    printf("vocab_size: %zu\n", vocab_size);
    printf("n_layers: %zu\n", n_layers);
    printf("n_heads: %zu\n", n_heads);
    printf("n_embd: %zu\n", n_embd);

    int *shape_buffer = (int*)mallocCheck(sizeof(int) * shape_header_size);
    freadCheck(shape_buffer, sizeof(int), shape_header_size, fp);

    int shape_index = 0;
    int num_param_tensors = 0;
    while (shape_index < shape_header_size) {
        int ndims = shape_buffer[shape_index];
        num_param_tensors += 1;
        shape_index += ndims + 1;
    }

    tensor_t **parameters = (tensor_t**)mallocCheck(sizeof(tensor_t*) * num_param_tensors);
    shape_index = 0;
    int param_index = 0;
    while (shape_index < shape_header_size) {
        int ndims = shape_buffer[shape_index];
        int shape[8];
        for (int i = 0; i < ndims; i++)
            shape[i] = shape_buffer[shape_index + 1 + i];
        
        parameters[param_index++] = tensor_load(fp, shape, ndims);
        shape_index += ndims + 1;
    }

    model->fast_load_state_dict(model, parameters);
    for (int i = 0; i < model->_num_param_tensors; i++)
        free_tensor(parameters[i]);
    free(parameters);

    free(shape_buffer);
    fcloseCheck(fp);
    return steps;
}


void save_model(const char *file_path, const gpt2_t *model, size_t steps) {
    if (model == NULL) {
        printf("Expected *model to be of type gpt2_t. Got NULL.\n");
        exit(1);
    }

    size_t max_block_size, vocab_size, n_layers, n_heads, n_embd;
    size_t shape_header_size = 0;

    max_block_size = model->block_size;
    vocab_size = model->vocab_size;
    n_layers = model->n_layers;
    n_heads = model->n_heads;
    n_embd = model->n_embd;

    tensor_t **parameters = model->parameters(model);
    for (int i = 0; i < model->_num_param_tensors; i++) {
        tensor_t *parameter = parameters[i];
        shape_header_size += parameter->ndims + 1; // (ndims, shape[0], shape[1], ..., shape[ndims - 1])
    }

    FILE *fp = fopenCheck(file_path, "wb");
    int *headers = (int*)mallocCheck(256 * sizeof(int));
    headers[0] = 20240415; // magic number
    headers[1] = max_block_size;
    headers[2] = vocab_size;
    headers[3] = n_layers;
    headers[4] = n_heads;
    headers[5] = n_embd;
    headers[6] = shape_header_size;
    headers[7] = steps;
    for (int i = 8; i < 256; i++)
        headers[i] = 0;

    int *shape_headers = (int*)mallocCheck(shape_header_size * sizeof(int));
    size_t shape_headers_index = 0;
    int parameter_index = 0;

    // Loops over all parameters in the model and stores 
    // the ndims and shape[j] in shape_headers
    while (shape_headers_index < shape_header_size && parameter_index < model->_num_param_tensors) {
        tensor_t *parameter = parameters[parameter_index++];
        shape_headers[shape_headers_index++] = parameter->ndims;
        for (int j = 0; j < parameter->ndims; j++)
            shape_headers[shape_headers_index++] = parameter->shape[j];
    }

    fwrite(headers, sizeof(int), 256, fp);
    fwrite(shape_headers, sizeof(int), shape_header_size, fp);

    // save model parameters
    for (int i = 0; i < model->_num_param_tensors; i++)
        tensor_save(fp, parameters[i]);

    free(headers);
    free(shape_headers);
    free(parameters); // we are only freeing the memory that holds pointers to parameters. We are not freeing the model parameters.
    fcloseCheck(fp);
}


int main() {

    GPT2Config_t gpt2_config;
    gpt2_config.block_size = 1024;
    gpt2_config.vocab_size = 50257;
    gpt2_config.n_embd = 768;
    gpt2_config.n_heads = 12;
    gpt2_config.n_layers = 12;

    gpt2_t *gpt = GPT2(&gpt2_config);
    int ckpt_steps = load_model(checkpoint_path, gpt);

    // create the dataloaders for training and validation
    dataloader_t *train_loader = DataLoader(tiny_shakespeare_train, batch_size, block_size);
    dataloader_t *val_loader = DataLoader(tiny_shakespeare_val, validation_batch_size, validation_block_size);

    // create optimizer
    adamW_t *optimizer = AdamW(
        gpt->parameters(gpt), 
        gpt->gradients(gpt), 
        gpt->_num_param_tensors, 
        lr, beta1, beta2, eps, weight_decay
    );

    // create loss_fn
    cross_entropy_loss_t *loss = CrossEntropyLoss();
    float best_training_loss = INFINITY;
    float best_validation_loss = INFINITY;

    printf("\n------------------------\n");
    printf("         Training       \n");
    printf("------------------------\n");

    struct timespec train_start, train_end, val_start, val_end;
    for (int step = 1; step <= training_steps; step++) {
        tensor_t *training_batch[2];
        train_loader->next(train_loader, training_batch);
        tensor_t *_x = training_batch[0], *_targets = training_batch[1];

        // we need to copy the tensors as the model always free's its inputs in backward pass
        // hence copying prevents us from losing the current batch's inputs and targets
        int inp_shape[2] = {batch_size, block_size};
        tensor_t *x = create_tensor(inp_shape, 2);
        tensor_t *targets = create_tensor(inp_shape, 2);

        tensor_copy(x, _x);
        tensor_copy(targets, _targets);

        // zero the gradients
        optimizer->zero_grad(optimizer);

        clock_gettime(CLOCK_MONOTONIC, &train_start);
        tensor_t *logits = gpt->forward(gpt, x);

        // calculate loss
        tensor_t *losses = loss->forward(loss, logits, targets);

        float training_mean_loss = 0.0f;
        for (int i = 0; i < losses->length; i++)
            training_mean_loss += losses->t[i];
        training_mean_loss /= losses->length;

        // backward pass
        for (int i = 0; i < losses->length; i++)
            losses->t[i] = 1.0f / losses->length;

        tensor_t *global_grad = loss->backward(loss, losses);
        global_grad = gpt->backward(gpt, global_grad);

        // update parameters
        optimizer->step(optimizer);

        clock_gettime(CLOCK_MONOTONIC, &train_end);
        double time_elapsed_s = (train_end.tv_sec - train_start.tv_sec) + (train_end.tv_nsec - train_start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, training_mean_loss, time_elapsed_s * 1000);

        if (training_mean_loss < best_training_loss)
            best_training_loss = training_mean_loss;

        free_tensor(logits);
        free_tensor(_x);
        free_tensor(_targets);

        // run validation every validation_interval
        if (step % validation_interval == 0 && val_loader) {
            printf("\nRunning validation\n");
            float mean_validation_loss = 0.0f;
            val_loader->reset(val_loader);
            int num_validation_steps = val_loader->len(val_loader);
            clock_gettime(CLOCK_MONOTONIC, &val_start);
            for (int val_step = 1; val_step <= num_validation_steps; val_step++)
            {
                tensor_t *validation_batch[2];
                val_loader->next(val_loader, validation_batch);
                tensor_t *val_x = validation_batch[0], *val_targets = validation_batch[1];

                tensor_t *val_logits = gpt->forward(gpt, val_x);
                tensor_t *val_losses = loss->forward(loss, val_logits, val_targets);

                float validation_batch_loss = 0.0f;
                for (int i = 0; i < val_losses->length; i++)
                    validation_batch_loss += val_losses->t[i];
                validation_batch_loss /= val_losses->length;
                mean_validation_loss += validation_batch_loss;

                gpt->free_cache(gpt);
                loss->free_cache(loss);
                free_tensor(val_losses);
                free_tensor(val_logits);
            }
            mean_validation_loss /= num_validation_steps;
            clock_gettime(CLOCK_MONOTONIC, &val_end);
            double val_time_elapsed_s = (val_end.tv_sec - val_start.tv_sec) + (val_end.tv_nsec - val_start.tv_nsec) / 1e9;
            printf("val loss: %f | val_batches: %d | validation took %f s\n", mean_validation_loss, num_validation_steps, val_time_elapsed_s);

            if (mean_validation_loss < best_validation_loss) {
                best_validation_loss = mean_validation_loss;
                save_model(c_checkpoint_path, gpt, step + ckpt_steps);
                printf("Model saved at %s.\n", c_checkpoint_path);
            }
            printf("\n");
        }
    }

    gpt->free_layer(gpt);
    optimizer->free_layer(optimizer);
    loss->free_layer(loss);
    train_loader->free_layer(train_loader);
    return 0;
}