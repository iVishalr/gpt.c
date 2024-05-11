#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "loss.h"

tensor_t *forward_cross_entropy_loss(cross_entropy_loss_t *loss, tensor_t *logits, tensor_t *targets) {

    if (loss == NULL) {
        printf("Expected required arugment *loss to be of type cross_entropy_loss ptr, but got NULL.\n");
        return NULL;
    }

    if (logits == NULL) {
        printf("Expected required argument *logits to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    if (targets == NULL) {
        printf("Expected required argument *targets to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    /*
        Explanation
        -----------

        In PyTorch, we would calculate the cross_entropy loss as follows:

        logits.shape = (B, T, V), where V is a list of probabilities that sum up to 1
        target.shape = (B, T) where T is a list of idx of next token 

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        In the above PyTorch code, logits.view(-1) collapses all dimensions except the last. Which 
        means logits is now (B * T, V). targets.view(-1) collapses all dimensions. Hence targets is
        now (B * T). 

        We now pick the value at targets[i] (idx <-- targets[i]); to obtain the correct class value. 
        Using this idx, we get the value at V[i]th row & V_i[idx] of logits.

        loss[i] <--- -log(V_i[idx])
    */

    int B, T, C;
    B = logits->shape[0];
    T = logits->shape[1];
    C = logits->shape[2];
    tensor_t *probs = loss->softmax->forward(loss->softmax, logits);

    tensor_t *out = zeros(targets->shape, targets->ndims);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *probs_bt = probs->t + b * T * C + t * C;
            int ix = (int)targets->t[b * T + t];
            out->t[b * T + t] = -logf(probs_bt[ix]);
        }
    }
    
    loss->cache[0] = probs; 
    loss->cache[1] = targets;
    return out;
}

tensor_t *backward_cross_entropy_loss(cross_entropy_loss_t *loss, tensor_t *global_grad) {

    if (loss == NULL) {
        printf("Expected required arugment *loss to be of type cross_entropy_loss_t ptr, but got NULL.\n");
        return NULL;
    }

    if (global_grad == NULL) {
        printf("Expected required argument *global_grad to be of type tensor_t ptr, but got NULL.\n");
        return NULL;
    }

    /*
        Explanation
        -----------

        global_grad: (B, T)
        loss->cache[0]: (B, T, V)
        loss->cache[1]: (B, T)

        We need to calculate gradients wrt logits that was passed to the forward function

        out[j] = { prob[j] - 1 * global_grad iif j == ix }, where ix = targets[i]
                 { prob[j] - 0 * global_grad iif j != ix }

        out: (B, T, V) # B, T will be collapsed
        prob / logits: (B, T, V) # B, T will be collapsed

    */

    int B, T, C;
    tensor_t *probs = loss->cache[0];
    tensor_t *targets = loss->cache[1];

    B = probs->shape[0];
    T = probs->shape[1];
    C = probs->shape[2];

    tensor_t *out = zeros(probs->shape, probs->ndims);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float *out_bt = out->t + b * T * C + t * C;
            float *prob_bt = probs->t + b * T * C + t * C;
            float dloss = global_grad->t[b * T + t];
            int ix = (int)targets->t[b * T + t]; 

            for (int i = 0; i < C; i++) {
                float p = prob_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                out_bt[i] += (p - indicator) * dloss;
            }
        }
    }

    free_tensor(global_grad);
    free_tensor(loss->cache[0]);
    free_tensor(loss->cache[1]);
    global_grad = NULL;
    loss->cache[0] = NULL;
    loss->cache[1] = NULL;
    return out;
}

int num_parameters_cross_entropy_loss(const cross_entropy_loss_t *loss) {
    return 0;
}

void description_cross_entropy_loss(const cross_entropy_loss_t *loss) {
    printf("CrossEntropyLoss()\n\n");
}

void free_layer_cross_entropy_loss(cross_entropy_loss_t *loss) {
    if (loss == NULL)
        return;

    loss->softmax->free_layer(loss->softmax);
    free_tensor(loss->cache[0]);
    free_tensor(loss->cache[1]);
    free(loss);
}

cross_entropy_loss_t *CrossEntropyLoss() {
    cross_entropy_loss_t *loss = (cross_entropy_loss_t *)malloc(sizeof(cross_entropy_loss_t));
    loss->softmax = Softmax();
    loss->cache[0] = NULL;
    loss->cache[1] = NULL;
    loss->forward = forward_cross_entropy_loss;
    loss->backward = backward_cross_entropy_loss;
    loss->description = description_cross_entropy_loss;
    loss->num_parameters = num_parameters_cross_entropy_loss;
    loss->free_layer = free_layer_cross_entropy_loss;
    return loss;
}