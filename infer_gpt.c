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
#include "tokenizer.h"
#include "utils.h"

// define command line options
const char *argp_program_version = "infer_gpt version 1.0";
static char *doc = "Infers from a GPT2 model";
static struct argp_option options[] = {
    // model settings
    {"load-checkpoint", 201, "LOAD_CHECKPOINT_PATH", 0, "Path to C model checkpoint to load the model. (Required)"},
    {"tokenizer", 202, "TOKENIZER", 0, "Path to tokenizer checkpoint. (Required)"},
    {"prompt", 203, "PROMPT", 0, "Prompt tokens to the model to kick off autoregressive prediction. (Required)"},
    {"max-tokens", 204, "MAX_TOKENS", OPTION_ARG_OPTIONAL, "Max number of tokens to generate. Default: 1024"},
    {"temperature", 205, "TEMPERATURE", OPTION_ARG_OPTIONAL, "Temperature to use during generation. Default: 1.0f"},
    {"rand-seed", 206, "RAND_SEED", OPTION_ARG_OPTIONAL, "Seed to initialize random generator. Default: 1337"},
    {"interactive", 207, "INTERACTIVE", OPTION_ARG_OPTIONAL, "Set this flag to enable live token generation. Default: disabled"},
    {0}
};


struct arguments {
    char *load_checkpoint;
    char *tokenizer_checkpoint;
    int *prompt;
    int num_init_tokens;
    int max_tokens;
    float temperature;
    int rand_seed;
    int interactive;
};


static void init_arguments(struct arguments *args) {
    if (!args) return;
    args->load_checkpoint = NULL;
    args->tokenizer_checkpoint = NULL;
    args->prompt = NULL;
    args->max_tokens = 1024;
    args->num_init_tokens = 0;
    args->temperature = 1.0f;
    args->rand_seed = 1337;
    args->interactive = 0;
}


// Function to trim leading and trailing whitespace
char* trim_whitespace(char* str) {
    char* end;

    // Trim leading space
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end+1) = 0;

    return str;
}

static error_t parse_options(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = state->input;
    char *ext = NULL;
    switch(key) {
        case 201:
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
            arguments->load_checkpoint = arg;
            break;
        case 202:
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
            arguments->tokenizer_checkpoint = arg;
            break;
        case 203:
            arg = trim_whitespace(arg);
            int prompt_len = strlen(arg);
            if (arg[0] == '[' && arg[prompt_len - 1] == ']') {
                arg++;
                arg[prompt_len - 2] = '\0';
            }

            // count number of tokens in the prompt
            int num_tokens = 1;
            for (char *i = arg; *i!='\0'; i++) {
                if (*i == ',') num_tokens++;
            }

            // allocate memory to prompt int array
            int *prompt_tokens = (int *)mallocCheck(sizeof(int) * num_tokens);
            int index = 0;
            for (char *p = strtok(arg, ","); p != NULL && index < num_tokens; p = strtok(NULL, ",")) {
                if (*p == ' ') p++;
                prompt_tokens[index++] = atoi(p);
            }
            arguments->prompt = prompt_tokens;
            arguments->num_init_tokens = num_tokens;
            break;
        case 204:
            if (arg != NULL)
                arguments->max_tokens = atoi(arg);
            break;
        case 205:
            if (arg != NULL)
                arguments->temperature = atof(arg);
            break;
        case 206:
            if (arg != NULL)
                arguments->rand_seed = atoi(arg);
            break;
        case 207:
            arguments->interactive = 1;
            break;
        case ARGP_KEY_ARG:
            return 0;
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}


static struct argp argp = {options, parse_options, 0, 0};


gpt2_t* load_model(const char *file_path) {
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

    GPT2Config_t gpt2_config;
    gpt2_config.block_size = max_block_size;
    gpt2_config.vocab_size = vocab_size;
    gpt2_config.n_embd = n_embd;
    gpt2_config.n_heads = n_heads;
    gpt2_config.n_layers = n_layers;

    gpt2_t *model = GPT2(&gpt2_config);

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
        
        parameters[param_index++] = tensor_load(fp, shape, ndims, CPU);
        shape_index += ndims + 1;
    }

    model->fast_load_state_dict(model, parameters);
    for (int i = 0; i < model->_num_param_tensors; i++)
        free_tensor(parameters[i]);
    free(parameters);

    free(shape_buffer);
    fcloseCheck(fp);
    return model;
}


unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}


int main(int argc, char **argv) {
    struct arguments inference_args;
    init_arguments(&inference_args);

    // parse commandline args
    if (argp_parse(&argp, argc, argv, 0, 0, &inference_args) != 0)
        return 1;

    gpt2_t *gpt = load_model(inference_args.load_checkpoint);
    if (inference_args.num_init_tokens > gpt->block_size) {
        printf(
            "Error: Prompt length %d exceeds model's block_size %d\n", 
            inference_args.num_init_tokens, 
            gpt->block_size
        );
        free(inference_args.prompt);
        exit(1);
    }

    // create Tokenizer
    tokenizer_t *tokenizer = Tokenizer(inference_args.tokenizer_checkpoint);
    uint64_t rng_state = inference_args.rand_seed;
    device_t device = CPU;

    int total_tokens = inference_args.num_init_tokens + inference_args.max_tokens;
    int block_size_multiple = total_tokens / gpt->block_size;
    block_size_multiple = total_tokens % gpt->block_size == 0 ? block_size_multiple : block_size_multiple + 1;
    
    int input_shape[2] = {1, gpt->block_size * block_size_multiple};
    tensor_t *X = fill(input_shape, 2, 50256.0f, device);

    // We need to hack the tensor here because model can only process block_size
    // tokens at a time. If max_tokens + num_init_tokens is greater than block_size,
    // we will need to dequeue the tokens from the start of the tensor and enqueue
    // the new token at the end. Basically, implementing a sliding window of size block_size.
    
    // However, we want to keep track of all the tokens model has seen. This can be 
    // done by allocating a large enough tensor to accomodate all tokens but send 
    // only block_size tokens to the model at each step.

    // The tensor will be initialized with tokenizer.eot_token at all positions and
    // the tokens from the prompt will be applied.
    
    for (int i = 0; i < inference_args.num_init_tokens; i++)
        X->t[i] = (float)inference_args.prompt[i];

    int start_index = 0;
    struct timespec inference_start, inference_end;
    clock_gettime(CLOCK_MONOTONIC, &inference_start);

    printf("Starting Inference\n\n");

    for (int i = inference_args.num_init_tokens; i < total_tokens; i++) {
        int window_input_shape[2] = {1, gpt->block_size};
        tensor_t *window_input = create_tensor(window_input_shape, 2, device);

        start_index += i < gpt->block_size ? 0 : 1;
        memcpy(window_input->t, X->t + start_index, gpt->block_size * sizeof(float));

        tensor_t *logits = gpt->forward(gpt, window_input);
        gpt->free_cache(gpt); // frees up window_input and other cached tensors

        // we only care about the (i-1)th prediction 
        // pluck out logits[:, [i-1], :]
        int logits_offset = i < gpt->block_size ? i - 1 : gpt->block_size - 1;

        float *logits_last_idx = logits->t + (logits_offset) * gpt->vocab_size;

        // scale by temperature
        for (int j = 0; j < gpt->vocab_size; j++)
            logits_last_idx[j] = logits_last_idx[j] / inference_args.temperature;
        
        // calculate softmax
        float maxval = -INFINITY;
        for (int j = 0; j < gpt->vocab_size; j++)
            if (logits_last_idx[j] > maxval)
                maxval = logits_last_idx[j];
        
        float sum = 0.0f;
        for (int j = 0; j < gpt->vocab_size; j++) {
            logits_last_idx[j] = expf(logits_last_idx[j] - maxval);
            sum += logits_last_idx[j];
        }
        for (int j = 0; j < gpt->vocab_size; j++)
            logits_last_idx[j] /= sum;

        // sample
        float coin = random_f32(&rng_state);
        int next_token = sample_mult(logits_last_idx, gpt->vocab_size, coin);
        X->t[i] = (float)next_token;
        
        if (inference_args.interactive == 1) {
            safe_printf(tokenizer->decode(tokenizer, next_token));
            fflush(stdout);
        }

        free_tensor(logits);
        logits = NULL;
        logits_last_idx = NULL;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &inference_end);
    double inference_time_s = (inference_end.tv_sec - inference_start.tv_sec) + (inference_end.tv_nsec - inference_start.tv_nsec) / 1e9;

    if (inference_args.interactive == 0) {
        uint32_t *tokens = (uint32_t *)mallocCheck(sizeof(uint32_t) * total_tokens);
        for (int i = 0; i < total_tokens; i++)
            tokens[i] = (uint32_t)X->t[i];
        
        uint32_t length = tokenizer->decode_length(tokenizer, tokens, total_tokens);
        char *dest = (char *)malloc(sizeof(char) * length + 1);
        tokenizer->decode_tokens(tokenizer, tokens, total_tokens, dest, length + 1);
        printf("%s", dest);
        fflush(stdout);
        free(tokens);
        free(dest);
    }

    printf("\n\nInference Time: %.4f seconds | tokens/s: %.2f\n", inference_time_s, (total_tokens - inference_args.num_init_tokens) / inference_time_s);

    free(inference_args.prompt);
    free_tensor(X);
    gpt->free_layer(gpt);
    tokenizer->free_layer(tokenizer);
    return 0;
}