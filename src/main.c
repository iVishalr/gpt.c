#include <stdio.h>
#include <time.h>

#include "transformer.h"
#include "optim.h"
#include "dataloader.h"

#define True 1
#define False 0

// main training loop
const char *tiny_shakespeare_train = "data/tiny_shakespeare/tiny_shakespeare_train.bin";
const char *tiny_shakespeare_val = "data/tiny_shakespeare/tiny_shakespeare_val.bin";

const int batch_size = 8;
const int block_size = 128;
const int training_steps = 100;
const float lr = 1e-4f;
const float beta1 = 0.9f;
const float beta2 = 0.999f;
const float eps = 1e-8f;
const float weight_decay = 0.00f;

int main() {

    GPT2Config_t gpt2_config;
    gpt2_config.block_size = 1024;
    gpt2_config.vocab_size = 50257;
    gpt2_config.n_embd = 768;
    gpt2_config.n_heads = 12;
    gpt2_config.n_layers = 12;

    gpt2_t *gpt = GPT2(&gpt2_config);
    gpt->description(gpt);
    // create the dataloaders for training and validation
    dataloader_t *train_loader = DataLoader(tiny_shakespeare_train, batch_size, block_size);
    // dataloader_t *val_loader = DataLoader(tiny_shakespeare_val, batch_size, block_size);

    // create optimizer
    adamW_t *optimizer = AdamW(
        gpt->parameters(gpt), 
        gpt->gradients(gpt), 
        gpt->_num_param_tensors, 
        lr, beta1, beta2, eps, weight_decay
    );

    // create loss_fn
    cross_entropy_loss_t *loss = CrossEntropyLoss();

    struct timespec start, end;
    for (int step = 1; step <= training_steps; step++) {
        tensor_t *batch[2];
        train_loader->next(train_loader, batch);
        tensor_t *_x = batch[0], *_targets = batch[1];

        // we need to copy the tensors as the model always free's its inputs in backward pass
        // hence copying prevents us from losing the current batch's inputs and targets
        int inp_shape[2] = {batch_size, block_size};
        tensor_t *x = create_tensor(inp_shape, 2);
        tensor_t *targets = create_tensor(inp_shape, 2);

        tensor_copy(x, _x);
        tensor_copy(targets, _targets);

        // zero the gradients
        optimizer->zero_grad(optimizer);

        clock_gettime(CLOCK_MONOTONIC, &start);
        tensor_t *logits = gpt->forward(gpt, x);

        // calculate loss
        tensor_t *losses = loss->forward(loss, logits, targets);

        float mean_loss = 0.0f;
        for (int i = 0; i < losses->length; i++)
            mean_loss += losses->t[i];
        mean_loss /= losses->length;

        // backward pass
        for (int i = 0; i < losses->length; i++)
            losses->t[i] = 1.0f / losses->length;

        tensor_t *global_grad = loss->backward(loss, losses);
        global_grad = gpt->backward(gpt, global_grad);

        // update parameters
        optimizer->step(optimizer);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, mean_loss, time_elapsed_s * 1000);

        free_tensor(_x);
        free_tensor(_targets);
    }

    gpt->free_layer(gpt);
    optimizer->free_layer(optimizer);
    loss->free_layer(loss);
    train_loader->free_layer(train_loader);
    return 0;
}