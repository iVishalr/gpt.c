#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <argp.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "transformer.h"
#include "optim.h"
#include "dataloader.h"
#include "utils.h"

// define command line options
const char *argp_program_version = "train_gpt version 1.0";
static char *doc = "Trains a GPT2 model";
static struct argp_option options[] = {
    // dataloader settings
    {"train-data", 101, "TRAIN_DATA_PATH", 0, "Path to training data."},
    {"val-data", 102, "VAL_DATA_PATH", OPTION_ARG_OPTIONAL, "Path to validation data. Default: None"},

    // training settings
    {"max-epochs", 201, "MAX_EPOCHS", OPTION_ARG_OPTIONAL, "Number of epochs to train the model. Default: 10"},
    {"batch-size", 202, "BATCH_SIZE", OPTION_ARG_OPTIONAL, "Batch size to use for training the model. Default: 8"},
    {"block-size", 203, "BLOCK_SIZE", OPTION_ARG_OPTIONAL, "Block size to use for Dataloader for training the model. Default: 128"},
    {"log-dir", 204, "LOG_DIR", OPTION_ARG_OPTIONAL, "Path to log directory to store checkpoints. Default: 'logs/'"},
    {"output", 205, "OUTPUT", OPTION_ARG_OPTIONAL, "Name of the model checkpoint. Default: 'checkpoint'"},
    {"load-checkpoint", 206, "LOAD_CHECKPOINT_PATH", OPTION_ARG_OPTIONAL, "Path to C model checkpoint to load the model from. Default: None"},

    // validation settings
    {"val-batch-size", 301, "VAL_BATCH_SIZE", OPTION_ARG_OPTIONAL, "Batch size to use for validation. Default: 8"},
    {"val-block-size", 302, "VAL_BLOCK_SIZE", OPTION_ARG_OPTIONAL, "Block size to use for validation. Default: 128"},
    {"val-interval", 303, "VAL_INTERVAL", OPTION_ARG_OPTIONAL, "Perform validation after every 'x' epochs. Default: 1"},

    // optimizer settings
    {"lr", 401, "OPTIMIZER_LR", OPTION_ARG_OPTIONAL, "Learning rate to use for optimization. Default: 3e-4"},
    {"weight-decay", 402, "OPTIMIZER_WEIGHT_DECAY", OPTION_ARG_OPTIONAL, "Weight decay to use for optimization. Default: 0.00"},
    {"beta1", 403, "OPTIMIZER_BETA1", OPTION_ARG_OPTIONAL, "Beta1 to use for optimization. Default: 0.9"},
    {"beta2", 404, "OPTIMIZER_BETA2", OPTION_ARG_OPTIONAL, "Beta2 to use for optimization. Default: 0.99"},
    {"eps", 405, "OPTIMIZER_EPS", OPTION_ARG_OPTIONAL, "Epsilon value to use for optimization. Default: 1e-8"},
    {0}
};

struct arguments {
    // dataloader settings
    char *train_data;
    char *val_data;

    // training settings
    int max_epochs;
    int batch_size;
    int block_size;
    char *log_dir;
    char *output;
    char *load_checkpoint;

    // validation settings
    int validation_batch_size;
    int validation_block_size;
    int validation_interval;

    // optimizer settings
    float lr;
    float weight_decay;
    float beta1;
    float beta2;
    float eps;
};


static void init_arguments(struct arguments *args) {
    if (!args) return;
    args->train_data = NULL;
    args->val_data = NULL;

    args->max_epochs = 10;
    args->batch_size = 8;
    args->block_size = 128;
    args->log_dir = "logs";
    args->output = "checkpoint";
    args->load_checkpoint = NULL;

    args->validation_batch_size = 8;
    args->validation_block_size = 128;
    args->validation_interval = 1;
    
    args->lr = 3e-4f;
    args->weight_decay = 0.0f;
    args->beta1 = 0.9f;
    args->beta2 = 0.99f;
    args->eps = 1e-8f;
}


static error_t parse_options(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    char *ext = NULL;
    switch (key) {
        case 101:
            if (!(access(arg, F_OK) == 0))
                argp_failure(state, 1, ENOENT, "%s", arg);
            if (!(access(arg, R_OK) == 0))
                argp_failure(state, 1, EACCES, "An error occured when opening the file '%s'", arg);
            ext = strrchr(arg, '.');
            if (!ext)
                argp_failure(state, 1, 0, "FileExtensionError: Expected the file '%s' to have '.bin' extension. Got NULL", arg);
            ext += 1;
            if (strcmp(ext, "bin") != 0)
                argp_failure(state, 1, 0, "FileExtensionError: Expected the file '%s' to have '.bin' extension. Got '%s'", arg, ext);
            arguments->train_data = arg;
            break;
        case 102:
            if (arg == NULL) {
                arguments->val_data = NULL;
                break;
            }
            if (!(access(arg, F_OK) == 0))
                argp_failure(state, 1, ENOENT, "%s", arg);
            if (!(access(arg, R_OK) == 0))
                argp_failure(state, 1, EACCES, "An error occured when opening the file %s", arg);
            ext = strrchr(arg, '.');
            if (!ext)
                argp_failure(state, 1, 0, "FileExtensionError: Expected the file '%s' to have '.bin' extension. Got NULL", arg);
            ext += 1;
            if (strcmp(ext, "bin") != 0)
                argp_failure(state, 1, 0, "FileExtensionError: Expected the file '%s' to have '.bin' extension. Got '%s'", arg, ext);
            arguments->val_data = arg;
            break;

        case 201:
            if (arg != NULL) arguments->max_epochs = atoi(arg);
            break;
        case 202:
            if (arg != NULL) arguments->batch_size = atoi(arg);
            break;
        case 203:
            if (arg != NULL) arguments->block_size = atoi(arg);
            break;
        case 204:
            if (arg != NULL) arguments->log_dir = arg;
            break;
        case 205:
            if (arg != NULL) arguments->output = arg;
            break;
        case 206:
            if (arg == NULL || !(access(arg, R_OK) == 0)) {
                argp_failure(state, 1, EACCES, "An error occured when opening the checkpoint file %s\n", arg);
                break;
            }
            arguments->load_checkpoint = arg;
            break;

        case 301:
            if (arg != NULL) arguments->validation_batch_size = atoi(arg);
            break;
        case 302:
            if (arg != NULL) arguments->validation_block_size = atoi(arg);
            break;
        case 303:
            if (arg != NULL) arguments->validation_interval = atoi(arg);
            break;

        case 401:
            if (arg != NULL) arguments->lr = atof(arg);
            break;
        case 402:
            if (arg != NULL) arguments->weight_decay = atof(arg);
            break;
        case 403:
            if (arg != NULL) arguments->beta1 = atof(arg);
            break;
        case 404:
            if (arg != NULL) arguments->beta2 = atof(arg);
            break;
        case 405:
            if (arg != NULL) arguments->eps = atof(arg);
            break;

        case ARGP_KEY_ARG:
            return 0;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}


static struct argp argp = {options, parse_options, 0, 0};


int load_model(const char *file_path, gpt2_t *model) {
    if (model == NULL) {
        printf("Expected *model to be of type gpt2_t. Got NULL.\n");
        exit(1);
    }

    FILE *fp = fopenCheck(file_path, "rb");

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

    char *keys[7] = {
        "max_block_size",
        "vocab_size",
        "n_layers",
        "n_heads",
        "n_embd",
        "checkpoint_path",
        "steps_trained"
    };
    char vals[7][1024];
    sprintf(vals[0], "%zu", max_block_size);
    sprintf(vals[1], "%zu", vocab_size);
    sprintf(vals[2], "%zu", n_layers);
    sprintf(vals[3], "%zu", n_heads);
    sprintf(vals[4], "%zu", n_embd);
    sprintf(vals[5], "%s", file_path);
    sprintf(vals[6], "%zu", steps);

    char *values[7] = { vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6] };

    printf("GPT2 Model Settings\n");
    print_table(keys, values, 7);
    printf("\n");

    int *shape_buffer = (int *)mallocCheck(sizeof(int) * shape_header_size);
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


int main(int argc, char **argv) {
    struct arguments training_config;
    init_arguments(&training_config);

    // parse commandline args
    if (argp_parse(&argp, argc, argv, 0, 0, &training_config) != 0)
        return 1;

    const char *train_data = training_config.train_data;
    const char *val_data = training_config.val_data;
    const int max_epochs = training_config.max_epochs;
    const int batch_size = training_config.batch_size;
    const int block_size = training_config.block_size;
    const char *load_checkpoint = training_config.load_checkpoint;
    const float lr = training_config.lr;

    struct stat log_dir_stat;
    int err = stat(training_config.log_dir, &log_dir_stat);
    if (err == -1) {
        if (ENOENT == errno) {
            printf("Creating logging directory: %s\n", training_config.log_dir);
            mkdir(training_config.log_dir, 0750);
        }
    } else {
        if (!S_ISDIR(log_dir_stat.st_mode)) {
            printf("Error: %s is not a directory.", training_config.log_dir);
            exit(1);
        }
    }

    char save_checkpoint_path[1024];
    sprintf(save_checkpoint_path, "%s/%s.bin", training_config.log_dir, training_config.output);

    // define GPT2 model
    GPT2Config_t gpt2_config;
    gpt2_config.block_size = 1024;
    gpt2_config.vocab_size = 50257;
    gpt2_config.n_embd = 768;
    gpt2_config.n_heads = 12;
    gpt2_config.n_layers = 12;

    gpt2_t *gpt = GPT2(&gpt2_config);
    int ckpt_steps = load_model(load_checkpoint, gpt);

    // create the dataloaders for training and validation
    dataloader_t *train_loader = DataLoader(
        training_config.train_data, 
        batch_size, 
        block_size
    );

    dataloader_t *val_loader = val_data ? DataLoader(
        training_config.val_data, 
        training_config.validation_batch_size, 
        training_config.validation_block_size
    ) : NULL;

    // create optimizer
    adamW_t *optimizer = AdamW(
        gpt->parameters(gpt),
        gpt->gradients(gpt),
        gpt->_num_param_tensors,
        lr, training_config.beta1, training_config.beta2, 
        training_config.eps, training_config.weight_decay
    );

    // create loss_fn
    cross_entropy_loss_t *loss = CrossEntropyLoss();

    char *keys[100] = {
        "train_data",
        "val_data",
        "log_dir",
        "save_checkpoint",
        "max_epochs",
        "train_batch_size",
        "train_block_size",
        "num_train_batches",
        "total_train_steps",
        "validation_enabled",
        "val_batch_size",
        "val_block_size",
        "val_interval",
        "num_val_batches",
        "lr",
        "weight_decay",
        "beta1",
        "beta2",
        "eps"
    };

    char vals[100][1024];
    sprintf(vals[0], "%s", training_config.train_data);
    sprintf(vals[1], "%s", training_config.val_data);
    sprintf(vals[2], "%s", training_config.log_dir);
    sprintf(vals[3], "%s", save_checkpoint_path);
    sprintf(vals[4], "%d", max_epochs);
    sprintf(vals[5], "%d", batch_size);
    sprintf(vals[6], "%d", block_size);
    sprintf(vals[7], "%d", train_loader->len(train_loader));
    sprintf(vals[8], "%d", train_loader->len(train_loader) * max_epochs);
    sprintf(vals[9], "%s", val_data ? "true" : "false");
    sprintf(vals[10], "%d", training_config.validation_batch_size);
    sprintf(vals[11], "%d", training_config.validation_block_size);
    sprintf(vals[12], "%d", training_config.validation_interval);
    sprintf(vals[13], "%d", val_loader ? val_loader->len(val_loader) : 0);
    sprintf(vals[14], "%.4e", lr);
    sprintf(vals[15], "%.4e", training_config.weight_decay);
    sprintf(vals[16], "%.4e", training_config.beta1);
    sprintf(vals[17], "%.4e", training_config.beta2);
    sprintf(vals[18], "%.4e", training_config.eps);

    int total_rows = 19;
    char *values[1024];
    for (int i = 0; i < total_rows; i++)
        values[i] = vals[i];

    printf("Training Settings\n");
    print_table(keys, values, 19);
    printf("\n");

    float best_training_loss = INFINITY;
    float best_validation_loss = INFINITY;

    struct timespec train_start, train_end, val_start, val_end;
    int total_training_steps = ckpt_steps;
    int training_steps = train_loader->len(train_loader);

    for (int epoch = 1; epoch <= max_epochs; epoch++) {
        for (int step = 1; step <= training_steps; step++) {
            total_training_steps += 1;

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
            printf("epoch: %d step: %d | train loss: %f lr: %.4e | took %.4f ms\n", epoch, step, training_mean_loss, lr, time_elapsed_s * 1000);

            if (training_mean_loss < best_training_loss)
                best_training_loss = training_mean_loss;

            free_tensor(logits);
            free_tensor(_x);
            free_tensor(_targets);
        }
        // run validation every validation_interval
        if (val_loader && epoch % training_config.validation_interval == 0) {
            printf("\nRunning validation\n");
            float mean_validation_loss = 0.0f;
            val_loader->reset(val_loader);
            int num_validation_steps = val_loader->len(val_loader);
            clock_gettime(CLOCK_MONOTONIC, &val_start);
            
            for (int val_step = 1; val_step <= num_validation_steps; val_step++) {
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
            printf("val loss: %f | val_batches: %d | validation took %.4f seconds\n", mean_validation_loss, num_validation_steps, val_time_elapsed_s);

            if (mean_validation_loss < best_validation_loss) {
                best_validation_loss = mean_validation_loss;
                save_model(save_checkpoint_path, gpt, total_training_steps);
                printf("Model saved at %s\n", save_checkpoint_path);
            }
            printf("\n");
        }
    }

    printf("\nTraining Statistics\n");
    printf("Best training loss: %f\n", best_training_loss);
    printf("Best validation loss: %f\n", best_validation_loss);
    printf("Latest model checkpoint: %s\n", save_checkpoint_path);

    gpt->free_layer(gpt);
    optimizer->free_layer(optimizer);
    loss->free_layer(loss);
    train_loader->free_layer(train_loader);
    return 0;
}