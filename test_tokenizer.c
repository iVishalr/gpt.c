#include "tokenizer.h"
#include <stdlib.h>

int main() {
    tokenizer_t *tokenizer = Tokenizer("./tokenizer.bin");

    safe_printf(tokenizer->decode(tokenizer, 50256));
    safe_printf(tokenizer->decode(tokenizer, 15496));
    safe_printf(tokenizer->decode(tokenizer, 2159));

    int tokens[3] = {50256, 15496, 2159};
    uint32_t length = tokenizer->decode_length(tokenizer, tokens, 3);
    printf("Length: %u\n", length); 
    char *dest = (char *)malloc(sizeof(char) * length + 1);
    tokenizer->decode_tokens(tokenizer, tokens, 3, dest, length + 1);
    printf("\n%s\n", dest);
    free(dest);
    tokenizer->free_layer(tokenizer);
    return 0;
}