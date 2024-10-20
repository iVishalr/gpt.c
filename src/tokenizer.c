#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "tokenizer.h"
#include "utils.h"

void free_layer_tokenizer(tokenizer_t *tokenizer);
char *decode_tokenizer(tokenizer_t *tokenizer, uint32_t token);
void decode_tokens_tokenizer(tokenizer_t *tokenizer, uint32_t *tokens, size_t n, char *dest, size_t dest_n);
uint32_t decode_length_tokenizer(tokenizer_t *tokenizer, uint32_t *tokens, size_t n);


tokenizer_t *Tokenizer(const char *filename) {
    FILE *file = fopenCheck(filename, "rb");

    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);

    CHECK_ERROR(header[0] != 20240415, "Bad magic number in %s file. Expected %d got %d\n", filename, 20240415, header[0]);

    tokenizer_t *tokenizer = (tokenizer_t *)mallocCheck(sizeof(tokenizer_t));
    tokenizer->vocab_size = header[1];
    tokenizer->eot_token = header[2];

    tokenizer->token_map = (char **)mallocCheck(sizeof(char *) * tokenizer->vocab_size);
    unsigned char length;
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        CHECK_ERROR(length == 0, "Expected every token to be atleast one character. Got 0\n");
        char *token_bytes = (char *)mallocCheck(sizeof(char) * length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';
        tokenizer->token_map[i] = token_bytes;
    }
    fcloseCheck(file);

    tokenizer->decode = decode_tokenizer;
    tokenizer->decode_tokens = decode_tokens_tokenizer;
    tokenizer->decode_length = decode_length_tokenizer;
    tokenizer->free_layer = free_layer_tokenizer;
    return tokenizer;
}


void free_layer_tokenizer(tokenizer_t *tokenizer) {
    if (tokenizer == NULL)
        return;

    for (uint32_t i = 0; i < tokenizer->vocab_size; i++)
        free(tokenizer->token_map[i]);
    free(tokenizer->token_map);
    free(tokenizer);
}


uint32_t decode_length_tokenizer(tokenizer_t *tokenizer, uint32_t *tokens, size_t n) {
    if (tokenizer == NULL || tokens == NULL)
        return 0;

    uint32_t length = 0;
    for (size_t i = 0; i < n; i++) {
        uint32_t token = tokens[i];
        length += strlen(tokenizer->token_map[token]);
    }
    return length;
}


char *decode_tokenizer(tokenizer_t *tokenizer, uint32_t token) {
    if (tokenizer == NULL)
        return NULL;
    if (token < tokenizer->vocab_size)
        return tokenizer->token_map[token];
    else {
        CHECK_ERROR(1, "Expected token to be less than vocab_size(%u). Got %u\n", tokenizer->vocab_size, token);
    }
}


void decode_tokens_tokenizer(tokenizer_t *tokenizer, uint32_t *tokens, size_t n, char *dest, size_t dest_n) {
    if (tokenizer == NULL || tokens == NULL || dest == NULL)
        return;
    
    size_t idx = 0;
    for (size_t i = 0; i < n; i++) { 
        uint32_t token = tokens[i];
        char *d = tokenizer->decode(tokenizer, token);
        while (*d != '\0' && idx < dest_n - 1) {
            dest[idx++] = *d;
            d++;
        }
    }
    dest[idx] = '\0';
}

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}