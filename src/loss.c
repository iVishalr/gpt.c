#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include "dispatch.h"
#include "loss.h"


tensor_t *forward_cross_entropy_loss(cross_entropy_loss_t *loss, tensor_t *logits, tensor_t *targets);
tensor_t *backward_cross_entropy_loss(cross_entropy_loss_t *loss, tensor_t *global_grad);
int num_parameters_cross_entropy_loss(const cross_entropy_loss_t *loss);
void description_cross_entropy_loss(const cross_entropy_loss_t *loss);
void free_layer_cross_entropy_loss(cross_entropy_loss_t *loss);
void free_cache_cross_entropy_loss(cross_entropy_loss_t *loss);
void to_cross_entropy_loss(cross_entropy_loss_t *loss, const device_t device);


// CrossEntropyLoss Class
cross_entropy_loss_t *CrossEntropyLoss() {
    cross_entropy_loss_t *loss = (cross_entropy_loss_t *)mallocCheck(sizeof(cross_entropy_loss_t));
    loss->cache[0] = NULL;
    loss->cache[1] = NULL;
    loss->forward = forward_cross_entropy_loss;
    loss->backward = backward_cross_entropy_loss;
    loss->description = description_cross_entropy_loss;
    loss->num_parameters = num_parameters_cross_entropy_loss;
    loss->free_layer = free_layer_cross_entropy_loss;
    loss->free_cache = free_cache_cross_entropy_loss;
    loss->to = to_cross_entropy_loss;
    return loss;
}

tensor_t *forward_cross_entropy_loss(cross_entropy_loss_t *loss, tensor_t *logits, tensor_t *targets) {

    CHECK_ERROR(loss == NULL, "Expected *loss to be a cross_entropy_loss_t pointer, but got NULL.");
    CHECK_ERROR(logits == NULL, "Expected *logits to be a tensor_t pointer, but got NULL.");
    CHECK_ERROR(targets == NULL, "Expected *targets to be a tensor_t pointer, but got NULL.");

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

    device_t device = logits->device;
    int out_shape[1] = {1};
    tensor_t *out = zeros(out_shape, 1, device);

    loss->cache[0] = create_tensor(logits->shape, logits->ndims, device);

    cross_entropy_forward_dispatch(logits, targets, loss->cache, out);

    loss->cache[1] = targets;
    return out;
}


tensor_t *backward_cross_entropy_loss(cross_entropy_loss_t *loss, tensor_t *global_grad) {

    CHECK_ERROR(loss == NULL, "Expected *loss to be a cross_entropy_loss_t pointer, but got NULL.");
    CHECK_ERROR(global_grad == NULL, "Expected *global_grad to be a tensor_t pointer, but got NULL.");

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

    device_t device = global_grad->device;
    tensor_t *log_softmax_output = loss->cache[0];
    tensor_t *dout = zeros(log_softmax_output->shape, log_softmax_output->ndims, device);

    cross_entropy_backward_dispatch(global_grad, (const tensor_t **)loss->cache, dout);

    free_tensor(global_grad);
    free_tensor(loss->cache[0]);
    free_tensor(loss->cache[1]);
    global_grad = NULL;
    loss->cache[0] = NULL;
    loss->cache[1] = NULL;
    return dout;
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

    free_tensor(loss->cache[0]);
    free_tensor(loss->cache[1]);
    free(loss);
}


void free_cache_cross_entropy_loss(cross_entropy_loss_t *loss) {
    if (loss == NULL)
        return;

    free_tensor(loss->cache[0]);
    free_tensor(loss->cache[1]);
    loss->cache[0] = NULL;
    loss->cache[1] = NULL;
}

void to_cross_entropy_loss(cross_entropy_loss_t *loss, const device_t device) {
    CHECK_ERROR(loss == NULL, "Expected *loss to be a cross_entropy_loss_t pointer, but got NULL.");

    if (loss->cache[0])
        loss->cache[0]->to(loss->cache[0], device);
    if (loss->cache[1])
        loss->cache[1]->to(loss->cache[1], device);
}