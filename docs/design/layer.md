# Design document for a Layer in gpt.c<!-- omit from toc -->

A transformer is composed of several layers, each performing a specific function. Each layer knows how it should behave during a forward pass and during a backward pass. In the forward pass, each layer applies a mathematical formula on the inputs and weights passed to the layer and computes the output. During the backward pass, the layer computes the local gradients to each of the inputs passed to the mathematical formula and multiplies it with gradient signals, coming from outside, to complete the chain rule.

This forward-backward API is followed in most deep learning frameworks. In gpt.c, the layers are designed such that they resemble PyTorch's way of creating a neural network layer.

## Table of Contents<!-- omit from toc -->
- [Layer Design](#layer-design)
  - [Creating layer structure](#creating-layer-structure)
  - [Writing layer definitions](#writing-layer-definitions)
    - [Forward Pass](#forward-pass)
    - [Backward Pass](#backward-pass)
    - [Parameters and Gradient APIs](#parameters-and-gradient-apis)
    - [Freeing Layer and Caches](#freeing-layer-and-caches)
    - [Loading Parameters](#loading-parameters)
  - [Nested Layers](#nested-layers)
    - [Nested Layer Structure](#nested-layer-structure)
    - [MLP and Block Layers](#mlp-and-block-layers)
    - [Forward Pass](#forward-pass-1)
    - [Backward Pass](#backward-pass-1)
    - [Parameters and Gradient APIs for Nested Layers](#parameters-and-gradient-apis-for-nested-layers)
    - [Freeing Layer and Caches](#freeing-layer-and-caches-1)
    - [Loading Parameters](#loading-parameters-1)
    - [Comparisons with PyTorch](#comparisons-with-pytorch)
- [Conclusion](#conclusion)


## Layer Design

This section provides a template for desigining a layer in gpt.c.

### Creating layer structure

We will start by creating a simple structure for our layer.

```C
typedef struct layer {
    // Define parameters of layer
    tensor_t *param1;
    tensor_t *param2;
    // more params ...

    // Define gradients of layer
    tensor_t *grad_param1;
    tensor_t *grad_param2;
    // more gradients ...

    // Define buffer to store inputs to be used in backward pass
    tensor_t *cache;
    // more buffers ...


    // Define layer APIs via function pointers (forward, backward)
    tensor_t *(*forward)(struct linear *, tensor_t *);
    tensor_t *(*backward)(struct linear *, tensor_t *);

    // Define APIs to get list of params or gradients
    tensor_t **(*parameters)(const struct linear *);
    tensor_t **(*gradients)(const struct linear *);

    // Define APIs for freeing layer
    void (*free_layer)(struct linear *);
    void (*free_cache)(struct linear *);

    // Define API to load params for the layer
    void (*load_state_dict)(struct linear *, tensor_t **);

    // Define layer specific attributes
    int attr1;
    int attr2;
    // more attrs ...

    int _num_param_tensors;

} layer_t;
```

Each layer will contain bunch of parameters (eg: Weights, biases, etc.). For each parameter defined in the structure, there should be a gradient tensor defined in the structure. Sometimes, layers need to cache the inputs passed to the layer during forward pass. This cached input will be used to compute the gradients during the backward pass. Any number of buffers can be defined in the structure for holding the required inputs needed during backward pass. 

Layer APIs or methods like forward(), backward() are defined as function pointers in the structure. These functions can be invoked as `layer->forward(layer, ...)`. This sort of mimicks syntax for calling class methods in Python (eg: layer.forward() or layer.backward()). Because the called function does not have access to the layer's structure, each function pointer should accept the layer struct as the first argument to the function. Any arguments can follow the layer struct argument in the function signature.

Since a layer can contain multiple parameters, it becomes difficult to access them iteratively. Hence, `layer->parameters(layer)` is defined in each layer that returns an array of pointers to the layer parameters. Similarly, there's a corresponding function for getting an array of pointers to the gradients. Note that these functions return an array that contains pointers to parameters / gradients. Hence, modifying this will modify the actual layer's parameters / gradients.

Since we are using C, we need to manually free the allocated objects. To do this, each layer will have `free_layer` and `free_cache` functions. `free_layer` will free the entire layer, including the current layer and other layers embedded in this structure. `free_cache` will only free the cache buffers used in the current layer and all the other layers embedded in this structure.

To support loading parameter values from outside the layer, `load_state_dict` function is provided. This function accepts an array of pointers to tensors as "state". Each layer will index the "state" to load the layer parameters.

The layer arguments are defined next. These arguments control the layer behaviour during forward and backward pass. Finally, there's `_num_param_tensors` attribute. This attribute stores the number of parameters present in the layer.

### Writing layer definitions

Now that we have looked how a layer structure is created, let's look at how to write the contents of the layer. We first import the header file that contains the structure for our layer. Then, we write the function definitions for the following:

- ```C
  // computes the forward pass
  tensor_t *forward_layer(layer_t *layer, tensor_t *x);
  ```
- ```C
  // computes the backward pass
  tensor_t *backward_layer(layer_t *layer, tensor_t *global_grad);
  ```
- ```C
  // returns a pointer to array of pointers to layer parameters
  tensor_t **parameters_layer(layer_t *layer);
  ```
- ```C
  // returns a pointer to array of pointers to layer gradients
  tensor_t **gradients_layer(layer_t *layer);
  ```
- ```C
  // frees all sublayers, allocated tensors and the current layer as well
  void free_layer_layer(layer_t *layer);
  ```
- ```C
  // frees only the tensors allocated for storing inputs
  void free_cache_layer(layer_t *layer);
  ```

We will write the main Layer "class". This function will create the layer object and initialize all the members in the layer's structure and return it. This is equivalent to performing `self.fc = nn.Linear(in_features=128, out_features=256, use_bias=True)` in PyTorch.

```C
// Layer Class
layer_t *Layer(int attr1, int attr2, ...) {
    layer_t *layer = (layer_t *)malloc(sizeof(layer_t));
    
    layer->attr1 = attr1;
    layer->attr2 = attr2;
    // more attrs ...
    
    layer->param1 = create_tensor(...);
    layer->param2 = create_tensor(...);
    layer->grad_param1 = zeros(...);
    layer->grad_param2 = zeros(...);
    
    layer->cache = NULL;
    // more caches ...

    // assign function pointers to functions
    layer->forward = forward_layer;
    layer->backward = backward_layer;
    layer->parameters = parameters_layer;
    layer->gradients = gradients_layer;
    layer->free_layer = free_layer;
    layer->free_cache = free_cache;
    layer->_num_param_tensors = <number of params in the layer>;
    return layer;
}
```

A top level function that wants to create this Layer can do it as follows:

```C
layer_t *layer1 = Layer(128, 256);
```

#### Forward Pass

The forward pass for each layer defines what the layer will compute when given inputs. The forward pass will take an input tensor as argument, perform some computation using the parameters defined in the layer and returns an ouput tensor. The layer will also cache the input if it requires access to the inputs during backward pass. The caching of input tensor **does not** create a copy of the tensor. This is done to reduce memory usage as input tensors, sometimes, can be very big, occupying several MBs!

The forward pass is written as follows:

```C
tensor_t *forward_layer(layer_t *layer, tensor_t *x) {
    if (x == NULL) return NULL;

    tensor_t *out = create_tensor(...);
    // dummy function to illustrate 
    // Use the input 'x' and layer's parameters to compute a 
    // function out = f(x, param1, param2, ...)
    // store the result in 'out' tensor.
    compute_operation(x, layer->param1, layer->param2, out);

    // cache the input tensor 'x'
    layer->cache = x;

    // return the output
    return out;
}
```

A top level function that wants to perform forward pass can do it as follows:

```C
tensor_t *input = ones(...);
tensor_t *output = layer->forward(layer, input);
```

This is similar to performing the following in PyTorch

```py
fc1 = nn.Linear(128, 256)
x = torch.ones(1, 128)
out = fc1(x)
```

*Note: The forward function of a layer will not free the input tensor.*

#### Backward Pass

The backward pass for each layer defines how the layer will compute the gradients for each one of the inputs passed to the forward function. The backward pass will take an input tensor or "global gradient", then compute the local gradient by differentiating the forward pass equation with each one of the inputs and then multiply each local gradient with the global gradient to complete the chain rule. After performing the backward pass, all parameters in the layer will have a gradient tensor populated, and a gradient tensor will be returned from the function. This gradient tensor is the gradient for the input tensor that was passed to the forward passs of the layer.

The backward pass will also free the inputs that were cached during the forward pass as it is no longer required to be maintained in memory.

The backward pass is written as follows:

```C
tensor_t *backward_layer(layer_t *layer, tensor_t *global_grad) {
    if (global_grad == NULL) return NULL;

    tensor_t dout, dparam1, dparam2;
    dout = zeros(...);
    dparam1 = zeros(...);
    dparam2 = zeros(...);
    // dummy functions to calculate local gradients
    compute_local_gradient(layer->cache, layer->param1, layer->param2, dout);
    compute_local_gradient(layer->cache, layer->param1, layer->param2, dparam1);
    compute_local_gradient(layer->cache, layer->param1, layer->param2, dparam2);

    multiply(dout, global_grad);
    multiply(dparam1, global_grad);
    multiply(dparam2, global_grad);

    layer->grad_param1 = dparam1;
    layer->grad_param2 = dparam2;

    // free the global grad tensor as it is no longer required
    free_tensor(global_grad);

    // free the cached input tensor as it is no longer required
    free_tensor(layer->cache);
    layer->cache = NULL;
    return dout;
}
```

A top level function that wants to perform backward pass can do it as follows:

```C
tensor_t *input = ones(...);
tensor_t *output = layer->forward(layer, input);

tensor_t *global_grad = ones(...);
tensor_t *dout = layer->backward(layer, global_grad);

// input tensor will be freed in backward()
// active tensors after backward() are: output, dout
```

This is similar to performing the following in PyTorch

```py
fc1 = nn.Linear(128, 256)
x = torch.ones(1, 128, requires_grad=True)
out = fc1(x)
out.backward(torch.ones_like(out))

print(out) # output in C equivalent
print(x.grad) # dout in C equivalent
print(fc1.weight.grad) # layer->grad_param1 in C equivalent
print(fc2.bias.grad) # layer->grad_param2 in C equivalent
```

*Note: The backward pass will free the cached input tensor as well as the global gradient that was passed as input*

#### Parameters and Gradient APIs

```C
tensor_t **parameters_layer(layer_t *layer) {
    tensor_t **params = (tensor_t **)malloc(sizeof(tensor_t *) * layer->_num_param_tensors);
    params[0] = layer->param1;
    params[1] = layer->param2;
    // more params ...

    return params;
}
```

```C
tensor_t **gradients_layer(layer_t *layer) {
    tensor_t **gradients = (tensor_t **)malloc(sizeof(tensor_t *) * layer->_num_param_tensors);
    gradients[0] = layer->grad_param1;
    gradients[1] = layer->grad_param2;
    // more gradients ...

    return gradients;
}
```

#### Freeing Layer and Caches

```C
void free_layer_layer(layer_t *layer) {
    if (layer == NULL) return;

    free_tensor(layer->cache);
    // more caches ...
    free_tensor(layer->param1);
    free_tensor(layer->param2);
    // more params ...
    free_tensor(layer->grad_param1);
    free_tensor(layer->grad_param2);
    // more gradients
    free(layer);
}
```

```C
void free_cache_layer(layer_t *layer) {
    if (layer == NULL) return;

    free_tensor(layer->cache);
    // more caches ...

    layer->cache = NULL; // important! Setting to NULL prevents from double freeing a tensor.
    // more caches ...
}
```

#### Loading Parameters

This is useful when we want to load a model from some pretrained weights. Each layer will have a method that accepts a "state" and loads (copy) the contents of the state to layer's parameters.

```C
void load_state_dict_layer(layer_t *layer, tensor_t **state) {
    copy(layer->param1, state[0]);
    copy(layer->param2, state[1]);
    // more params ...
}
```

That completes the necessary building blocks for creating a layer from scratch. Next we will see how this design makes it easier when a layer contains multiple layers.

### Nested Layers

In this section, we will look at how the building blocks shown in the previous section help to design layers that contain multiple other layers. Let's assume there are three basic layers: Linear, LayerNorm, GeLU. These three layers follow the building blocks shown in the previous section. Now, let's create two higher level layers: Block and MLP.

#### Nested Layer Structure

We will start with MLP layer. First, let's create the MLP structure.

```C
typedef struct mlp {
    gelu_t *gelu;
    linear_t *fc1, *fc2;

    // <usual layer function pointers go here>
    tensor_t *(*forward)(struct mlp *, tensor_t *);
    tensor_t *(*backward)(struct mlp *, tensor_t *);
    // more function pointers ...

    int in_features;
    int expansion_factor;
} mlp_t;
```

The above MLP layer uses two of the basic layers: Linear and GeLU. Since MLP layer itself doesn't have any parameters, there's no need to create params and grad_params tensors.

Now, let's create a structure for Block layer.

```C
typedef struct block {
    mlp_t *mlp1, *mlp2;
    layer_norm_t *ln1, *ln2;

    // <usual layer function pointers go here>
    tensor_t *(*forward)(struct block *, tensor_t *);
    tensor_t *(*backward)(struct block *, tensor_t *);
    // more function pointers ...

} block_t;
```

The above block layer uses two MLP layers and two LayerNorm layers.

#### MLP and Block Layers

Let's see how the class definitions for MLP and Block layers will look like.

```C
mlp_t *MLP(int in_features, int expansion_factor) {
    mlp_t *mlp = (mlp_t *)malloc(sizeof(mlp_t));
    mlp->fc1 = Linear(in_features, in_features * expansion_factor);
    mlp->gelu = GeLU();
    mlp->fc2 = Linear(in_features * expansion_factor, in_features);

    mlp->forward = forward_mlp;
    mlp->backward = backward_mlp;
    // more function pointers ...

    return mlp;
}
```

```C
block_t *Block(int in_features) {
    block_t *block = (block_t *)malloc(sizeof(block_t));
    block->mlp1 = MLP(in_features, 4);
    block->ln1 = LayerNorm(in_features);
    block->mlp2 = MLP(in_features, 4);
    block->ln2 = LayerNorm(in_features);

    block->forward = forward_block;
    block->backward = backward_block;
    // more function pointers ...

    return block;
}
```

#### Forward Pass

Let's see how the forward() for MLP layer looks like. Since MLP layer relies on other layers, we just need to call forward() on each of the sublayers.

```C
tensor_t *forward_mlp(mlp_t *mlp, tensor_t *x) {
    tensor_t *out;
    linear_t *fc1 = mlp->fc1;
    linear_t *fc2 = mlp->fc2;
    gelu_t *gelu = mlp->gelu;

    out = fc1->forward(fc1, x);
    out = gelu->forward(gelu, out);
    out = fc2->forward(fc2, out);
    return out;
}
```

That's it! 

Let's look at how the forward() for the Block layer looks like. Since the Block layer only relies on MLP and LayerNorm, we just need to call forward() on each of the sublayers.

```C
tensor_t *forward_block(block_t *block, tensor_t *x) {
    tensor_t *out;
    mlp_t *mlp1 = block->mlp1;
    mlp_t *mlp2 = block->mlp2;
    layer_norm_t *ln1 = block->ln1;
    layer_norm_t *ln2 = block->ln2;

    out = mlp1->forward(mlp1, x);
    out = ln1->forward(ln1, out);
    out = mlp2->forward(mlp2, out);
    out = ln2->forward(ln2, out);
    return out;
}
```

#### Backward Pass

Let's see how the backward() for MLP layer looks like. Since MLP layer relies on other layers, we just need to call backward() on each of the sublayers. Remember that the backward() frees any cached input tensors and the global gradient the function receives.

```C
tensor_t *backward_mlp(mlp_t *mlp, tensor_t *global_grad) {
    tensor_t *out = global_grad;
    linear_t *fc1 = mlp->fc1;
    linear_t *fc2 = mlp->fc2;
    gelu_t *gelu = mlp->gelu;

    out = fc1->backward(fc1, out);
    out = gelu->backward(gelu, out);
    out = fc2->backward(fc2, out);
    return out;
}
```

That's it! 

Let's look at how the backward() for the Block layer looks like. Since the Block layer only relies on MLP and LayerNorm, we just need to call backward() on each of the sublayers. Remember that the backward() frees any cached input tensors and the global gradient the function receives.

```C
tensor_t *backward_block(block_t *block, tensor_t *global_grad) {
    tensor_t *out = global_grad;
    mlp_t *mlp1 = block->mlp1;
    mlp_t *mlp2 = block->mlp2;
    layer_norm_t *ln1 = block->ln1;
    layer_norm_t *ln2 = block->ln2;

    out = mlp1->backward(mlp1, out);
    out = ln1->backward(ln1, out);
    out = mlp2->backward(mlp2, out);
    out = ln2->backward(ln2, out);
    return out;
}
```

#### Parameters and Gradient APIs for Nested Layers

```C
tensor_t **parameters_mlp(mlp_t *mlp) {
    tensor_t **params = (tensor_t **)malloc(sizeof(tensor_t *) * mlp->_num_param_tensors); // here _num_param_tensors will be the sum of all parameters in sublayers and parameters in mlp (if exists)
    
    tensor_t **fc1_params, **fc2_params;
    fc1_params = mlp->fc1->parameters(mlp->fc1);
    fc2_params = mlp->fc2->parameters(mlp->fc2);

    int index = 0;
    for (int i = 0; i < mlp->fc1->_num_param_tensors; i++)
        params[index++] = fc1_params[i];

    for (int i = 0; i < mlp->fc2->_num_param_tensors; i++)
        params[index++] = fc2_params[i];
    // more params ...

    return params;
}
```

```C
tensor_t **gradients_mlp(mlp_t *mlp) {
    tensor_t **gradients = (tensor_t **)malloc(sizeof(tensor_t *) * mlp->_num_param_tensors); // here _num_param_tensors will be the sum of all parameters in sublayers and parameters in mlp (if exists)
    
    tensor_t **fc1_gradients, **fc2_gradients;
    fc1_gradients = mlp->fc1->gradients(mlp->fc1);
    fc2_gradients = mlp->fc2->gradients(mlp->fc2);

    int index = 0;
    for (int i = 0; i < mlp->fc1->_num_param_tensors; i++)
        gradients[index++] = fc1_gradients[i];

    for (int i = 0; i < mlp->fc2->_num_param_tensors; i++)
        gradients[index++] = fc2_gradients[i];
    // more gradients ...

    return gradients;
}
```

#### Freeing Layer and Caches

```C
void free_layer_mlp(mlp_t *mlp) {
    if (mlp == NULL) return;

    linear_t *fc1 = mlp->fc1;
    linear_t *fc2 = mlp->fc2;
    gelu_t *gelu = mlp->gelu;

    fc1->free_layer(fc1);
    fc2->free_layer(fc2);
    gelu->free_layer(gelu);
    free(mlp);
}
```

```C
void free_cache_mlp(mlp_t *mlp) {
    if (mlp == NULL) return;

    linear_t *fc1 = mlp->fc1;
    linear_t *fc2 = mlp->fc2;
    gelu_t *gelu = mlp->gelu;

    fc1->free_cache(fc1);
    fc2->free_cache(fc2);
    gelu->free_cache(gelu);
}
```

#### Loading Parameters

This is useful when we want to load a model from some pretrained weights. When a layer consists of multiple sublayers, we pass the pointer to the start of sublayer's parameters.

```C
void load_state_dict_mlp(mlp_t *mlp, tensor_t **state) {
    linear_t *fc1 = mlp->fc1;
    linear_t *fc2 = mlp->fc2;
    gelu_t *gelu = mlp->gelu;
    
    fc1->load_state_dict(fc1, state);
    state += fc1->_num_param_tensors; // we have consumed params for fc1, move the state pointer
    fc2->load_state_dict(fc2, state);
    state += fc2->_num_param_tensors; // we have consumed params for fc2, move the state pointer
    // more params ...
}
```

#### Comparisons with PyTorch

In this section, we will see how similar it is to create layers in gpt.c compared to PyTorch.

1. Layer Class initialization

    <table>
    <tr>
    <th>gpt.c</th>
    <th>PyTorch</th>
    </tr>
    <tr>
    <td>
    
    ```C
    mlp_t *MLP(int in_features, int expansion_factor) {
        mlp_t *mlp = (mlp_t *)malloc(sizeof(mlp_t));
        mlp->fc1 = Linear(
            in_features, 
            in_features * expansion_factor
        );
        mlp->gelu = GeLU();
        mlp->fc2 = Linear(
            in_features * expansion_factor, 
            in_features
        );

        mlp->forward = forward_mlp;
        mlp->backward = backward_mlp;
        // more function pointers ...

        return mlp;
    }
    ```
    
    </td>
    <td>

    ```py
    class MLP(nn.Module):
        def __init__(self, 
            in_features, 
            expansion_factor
        ):
            self.fc1 = nn.Linear(
                in_features=in_features, 
                out_features=in_features * expansion_factor
            )
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(
                in_features=in_features * expansion_factor, 
                out_features=in_features
            )
    ```

    </td>
    </tr>
    </table>

2. Forward Pass

    <table>
    <tr>
    <th>gpt.c</th>
    <th>PyTorch</th>
    </tr>
    <tr>
    <td>
    
    ```C
    tensor_t *forward_mlp(mlp_t *mlp, tensor_t *x) {
        tensor_t *out;
        linear_t *fc1 = mlp->fc1;
        linear_t *fc2 = mlp->fc2;
        gelu_t *gelu = mlp->gelu;

        out = fc1->forward(fc1, x);
        out = gelu->forward(gelu, out);
        out = fc2->forward(fc2, out);
        return out;
    }
    ```
    
    </td>
    <td>

    ```py
    class MLP(nn.Module):
        def __init__(self, in_features, expansion_factor):
            ...
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.gelu(out)
            out = self.fc2(out)
            return out
    ```

    </td>
    </tr>
    </table>

Both look pretty similar :) 

## Conclusion

In this document, we have looked at how gpt.c designs layers. We looked at how the layer building blocks helps to build more advanced layers, all while keeping the overall look and feel like that of PyTorch. Now, you can go ahead and take a look at all the layers present in `src/` folder.