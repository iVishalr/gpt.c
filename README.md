# gpt.c<!-- omit from toc -->

<p align="center">
<img src="docs/images/banner.svg", alt="banner.svg"></img>
</p>

gpt.c is a simple C implementation of OpenAI's popular GPT-2 model. This project was originally inspired by Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c). However, llm.c has evolved into a great, but complex and slightly confusing, codebase over time. This project aims to build GPT-2 from scratch in C, keeping it as simple as possible while still achieving very good performance. At the time of writing, gpt.c matches PyTorch's speed on CPU. gpt.c uses OpenBLAS to accelerate matrix operations, which is essential for making the project practical without being extremely slow. To enhance understanding, most layers in gpt.c are designed in a way that makes building models very similar to doing so in PyTorch.

## Table of Contents<!-- omit from toc -->

- [Project Setup](#project-setup)
  - [Clone Project](#clone-project)
  - [Install dependencies](#install-dependencies)
  - [Compile Third Party libraries](#compile-third-party-libraries)
  - [Compiling gpt.c](#compiling-gptc)
- [Quick Start](#quick-start)
  - [Downloading datasets](#downloading-datasets)
  - [Training](#training)
  - [Inference](#inference)
  - [Finetuning with Pretrained Weights](#finetuning-with-pretrained-weights)
  - [Autocompletion using Pretrained Models](#autocompletion-using-pretrained-models)
- [Understanding gpt.c code](#understanding-gptc-code)
- [Contributing Guidelines](#contributing-guidelines)
- [TODO](#todo)


## Project Setup

### Clone Project

Clone the project along with submodules

```bash
git clone https://github.com/iVishalr/gpt.c.git
git submodule update --init --recursive
```

### Install dependencies

Install the required dependencies. This assumes that you already have python set up.

**Linux**

```bash
sudo apt install libomp-dev build-essential gcc gfortran make valgrind
pip install -r requirements.txt
```

**macOS**

_Note: Requires `brew` package manager to be installed._

```bash
brew install libomp
brew install argp-standalone
brew install gfortran
pip install -r requirements.txt
```

### Compile Third Party libraries

gpt.c relies on OpenBLAS for accelerating operations involving matrices. OpenBLAS library should be compiled before compiling gpt.c.

```bash
make third_party
```

This will take a while to compile depending on your system. You should not get any errors at any stage in the compilation. On successful compilation, you should see the following folders created:

```bash
third_party/OpenBLAS/include
third_party/OpenBLAS/lib
```

These folders contain the C header files and the openblas shared library which can be dynamically linked with other applications.

### Compiling gpt.c

Once the third party libraries are compiled, you can compile gpt.c using 

```bash
make
```

Supported compilers for compiling gpt.c include `gcc` (default) and `clang`. To use a specific compiler (eg: clang) use:

```bash
make CC=clang
```

_Note: Recommended compiler is `gcc`. `gcc` produces binaries that are 2x faster compared to `clang`. Reason is unknown yet._

_Note: On macOS, when compiling using `gcc` compiler, the compilation happens using `clang`. This is because, by default, the gcc binary is aliased to clang provided by Apple._

By default, the C files are compiled with all optimizations enabled. For debugging purposes, it is recommended to build with no optimizations and with debug symbols in place. You can produce the debug build using:

```bash
make BUILD=debug
```

Do note that using debug build will produce binaries that are an order of magnitude slower compared to the default compilation and should only be used for debugging purposes only.

Please make sure you execute the following command for the libraries to be loaded at runtime.

**Linux**

```bash
export LD_LIBRARY_PATH=third_party/OpenBLAS/lib:$LD_LIBRARY_PATH
```

**macOS**

```bash
BREW_PATH=$(brew --prefix)
export DYLD_LIBRARY_PATH=$(BREW_PATH)/opt/libomp/lib:$(BREW_PATH)/opt/argp-standalone/lib:third_party/OpenBLAS/lib:$DYLD_LIBRARY_PATH
```

Optionally, you can also add this statement in your `.bashrc` so that you don't have to execute it everytime you restart bash session.

## Quick Start

### Downloading datasets

gpt.c provides with three default datasets: Linar Algebra, Tiny Shakespeare and Tiny Stories. The dataset can be downloaded and processed by executing the python scripts present under `data/` as follows:

```bash
python3 data/tinyshakespeare.py
```

These python scripts downloads the dataset from internet and preprocesses them to make it suitable for gpt.c. During preprocessing, the data is tokenized using the GPT-2 tokenizer obtained from `tiktoken` and the tokens are written to a `.bin` file. The dataset will be split into train (90%) and validation (10%) splits and stored in `data/<dataset>/<dataset>_train.bin` and `data/<dataset>/<dataset>_val.bin` files respectively.

### Training

Once the datasets are prepared, we are ready to train a GPT2 model completely in C. Before that, we will need to create a initial model file using `model.py`. This model file will contain the untrained intial weights for the model that is organized in a way that gpt.c can recognize.

```bash
python3 model.py
```

This script creates a GPT2 model using PyTorch and dumps the model parameters into a binary file. This binary file will then be passed to the C training script to load the model weights from. By default, the above script creates the GPT2 124M parameter model with 12 layers, 12 heads, 1024 block size, 768 embedding dims and 50257 vocab size. You can override each of the model settings by passing them via commandline args as follows:

```bash
python3 model.py --block-size=128 --vocab-size=50257 --layers=2 --heads=12 --embd=128
```

It is recommended to use `--vocab-size=50257` as the datasets are processed using the GPT2 tokenizer from tiktoken. Take a look at other options exposed by the script using `python3 model.py -h`.

To kickoff the training, execute the following:

```bash
OPENBLAS_NUM_THREADS=6 ./train_gpt --train-data=data/tinyshakespeare/tinyshakespeare_train.bin --val-data=data/tinyshakespeare/tinyshakespeare_val.bin --max-epochs=2 --log-dir=model_checkpoints --output=gpt2 --batch-size=8 --block-size=128 --lr=3e-4 --load-checkpoint="./model/gpt2.bin"
```

Take a look at what each of the cmd options does by executing `./train_gpt -h`. 

It is highly recommended to set `OPENBLAS_NUM_THREADS` to the number of _physical_ cores on your machine. 

<details>

<summary>Training Output</summary>

```console
GPT2 Model Settings
+-----------------+------------------+
| Parameter       | Value            |
+-----------------+------------------+
| max_block_size  | 1024             |
| vocab_size      | 50257            |
| n_layers        | 12               |
| n_heads         | 12               |
| n_embd          | 768              |
| checkpoint_path | ./model/gpt2.bin |
| steps_trained   | 0                |
+-----------------+------------------+

Training Settings
+--------------------+------------------------------------------------+
| Parameter          | Value                                          |
+--------------------+------------------------------------------------+
| train_data         | data/tinyshakespeare/tinyshakespeare_train.bin |
| val_data           | data/tinyshakespeare/tinyshakespeare_val.bin   |
| log_dir            | model_checkpoints                              |
| save_checkpoint    | model_checkpoints/gpt2.bin                     |
| max_epochs         | 2                                              |
| train_batch_size   | 8                                              |
| train_block_size   | 128                                            |
| num_train_batches  | 297                                            |
| total_train_steps  | 594                                            |
| validation_enabled | true                                           |
| val_batch_size     | 8                                              |
| val_block_size     | 128                                            |
| val_interval       | 1                                              |
| num_val_batches    | 31                                             |
| lr                 | 3.0000e-04                                     |
| weight_decay       | 0.0000e+00                                     |
| beta1              | 9.0000e-01                                     |
| beta2              | 9.9000e-01                                     |
| eps                | 1.0000e-08                                     |
+--------------------+------------------------------------------------+

epoch: 1 step: 1 | train loss: 10.937562 lr: 3.0000e-04 | took 2888.1516 ms
epoch: 1 step: 2 | train loss: 9.798989 lr: 3.0000e-04 | took 2199.8180 ms
epoch: 1 step: 3 | train loss: 9.245797 lr: 3.0000e-04 | took 2026.6486 ms
epoch: 1 step: 4 | train loss: 8.937704 lr: 3.0000e-04 | took 2013.3016 ms
epoch: 1 step: 5 | train loss: 8.707121 lr: 3.0000e-04 | took 2007.4027 ms
epoch: 1 step: 6 | train loss: 8.705685 lr: 3.0000e-04 | took 2005.1602 ms
epoch: 1 step: 7 | train loss: 8.177757 lr: 3.0000e-04 | took 2035.2757 ms
epoch: 1 step: 8 | train loss: 8.172958 lr: 3.0000e-04 | took 2037.2050 ms
epoch: 1 step: 9 | train loss: 8.105190 lr: 3.0000e-04 | took 2035.6992 ms
.. (truncated) ..
Running validation
val loss: 6.727935 | val_batches: 31 | validation took 21.1506 seconds
Model saved at model_checkpoints/gpt2.bin
.. (truncated) ..

Training Statistics
Best training loss: 5.827348
Best validation loss: 6.707435
Latest model checkpoint: model_checkpoints/gpt2.bin
```

</details>

### Inference

When you have a trained model, you can also perform inference using it completely in C. Since gpt.c only understands tokens from GPT2 tokenizer, the user prompts must first be tokenized using the tokenizer to obtain the raw integer tokens. Additionally, the inference script also requires the tokenizer to be saved in a format that gpt.c understands. 

To save the GPT2 tokenizer, use the following:

```bash
python3 tokenizer.py -s
```

A `tokenizer.bin` file is created after executing the above command. This file will contain the weights of the GPT2 tokenizer as obtained from tiktoken. Additional options present in `tokenizer.py` can be viewed using `python3 tokenizer.py -h`.

Now, we need to encode the user's prompts before passing to `infer_gpt`. You can do so using the following:

```bash
PROMPT="ANTONIO"
TOKENS=$(python3 tokenizer.py -p "$PROMPT" -e)
echo $TOKENS
```

The generated tokens for the above PROMPT will look as follows:

```bash
[50256, 8643, 1340, 9399]
```

Now, we are ready to generate more tokens using `infer_gpt`.

```bash
OPENBLAS_NUM_THREADS=6 ./infer_gpt --load-checkpoint=model/gpt2.bin --tokenizer=tokenizer.bin --prompt="$TOKENS" --max-tokens=200 --interactive
```

<details>

<summary>Inference Output</summary>

```console
GPT2 Model Settings
+-----------------+----------------------------+
| Parameter       | Value                      |
+-----------------+----------------------------+
| max_block_size  | 1024                       |
| vocab_size      | 50257                      |
| n_layers        | 12                         |
| n_heads         | 12                         |
| n_embd          | 768                        |
| checkpoint_path | model_checkpoints/gpt2.bin |
| steps_trained   | 594                        |
+-----------------+----------------------------+

Starting Inference



 IRevThey I in they all age no meOf


 for
 soil o ware's,URte, him againORT
 wr: fou.
 led to'
 must'd bless strength presazed
kindra beatustomI is sweet?<|endoftext|> merelloTh,'s prevented wife<|endoftext|>

 thy meltseeds:.IO, gentleman, lastO hands them me
 good deerTh everTTo apartVery as one told
itent
:? the I d remembrance h now father noble beastOMous circled nature able
ce,
 forthare, oneB
As.
 sulph far,Thisation and
<|endoftext|>He wasI withx, word put now, thouHas way
 firstL fail night headch. outward oneI.-
tis;ine'elloIS sits list,.PET
 horse why,ORT For our,, bad
 and wand indeed son what
 welcome deservedSweetONE but, bears bring tra have story itL much but tomorrow

Inference Time: 251.6741 seconds | tokens/s: 0.79
```

I think it is pretty decent for a model that is trained only for two epochs or 594 steps. We'll see how this improves when we switch to a pretrained model and perform fine tuning for a few epochs.

</details>

### Finetuning with Pretrained Weights

Training a GPT2 from scratch may not be practical always. To speed up training, you can download the pretrained weights for GPT2 and then fine tune on your dataset. In gpt.c, you can use the `model.py` script to download pretrained weights.

To download the pretrained GPT2 weights, use the following:

_Note: Caution, large files will be downloaded for gpt2-large and gpt2-xl._

```bash
python3 model.py --from-pretrained gpt2 --name gpt2
python3 model.py --from-pretrained gpt2-medium --name gpt2-medium
python3 model.py --from-pretrained gpt2-large --name gpt2-large
python3 model.py --from-pretrained gpt2-xl --name gpt2-xl
```

Finetune the 124M model (gpt2) by training it on our custom dataset.

```bash
OPENBLAS_NUM_THREADS=6 ./train_gpt --train-data=data/tinyshakespeare/tinyshakespeare_train.bin --val-data=data/tinyshakespeare/tinyshakespeare_val.bin --max-epochs=2 --log-dir=model_checkpoints --output=gpt2 --batch-size=8 --block-size=128 --lr=3e-4 --load-checkpoint="./model/gpt2.bin"
```

<details>

<summary>Training Output</summary>

```console
GPT2 Model Settings
+-----------------+------------------+
| Parameter       | Value            |
+-----------------+------------------+
| max_block_size  | 1024             |
| vocab_size      | 50257            |
| n_layers        | 12               |
| n_heads         | 12               |
| n_embd          | 768              |
| checkpoint_path | ./model/gpt2.bin |
| steps_trained   | 0                |
+-----------------+------------------+

Training Settings
+--------------------+------------------------------------------------+
| Parameter          | Value                                          |
+--------------------+------------------------------------------------+
| train_data         | data/tinyshakespeare/tinyshakespeare_train.bin |
| val_data           | data/tinyshakespeare/tinyshakespeare_val.bin   |
| log_dir            | model_checkpoints                              |
| save_checkpoint    | model_checkpoints/gpt2.bin                     |
| max_epochs         | 2                                              |
| train_batch_size   | 8                                              |
| train_block_size   | 128                                            |
| num_train_batches  | 297                                            |
| total_train_steps  | 594                                            |
| validation_enabled | true                                           |
| val_batch_size     | 8                                              |
| val_block_size     | 128                                            |
| val_interval       | 1                                              |
| num_val_batches    | 31                                             |
| lr                 | 3.0000e-04                                     |
| weight_decay       | 0.0000e+00                                     |
| beta1              | 9.0000e-01                                     |
| beta2              | 9.9000e-01                                     |
| eps                | 1.0000e-08                                     |
+--------------------+------------------------------------------------+

epoch: 1 step: 1 | train loss: 4.712466 lr: 3.0000e-04 | took 2864.6513 ms
epoch: 1 step: 2 | train loss: 4.580047 lr: 3.0000e-04 | took 2142.8506 ms
epoch: 1 step: 3 | train loss: 4.273422 lr: 3.0000e-04 | took 1949.3644 ms
epoch: 1 step: 4 | train loss: 4.167133 lr: 3.0000e-04 | took 1961.8056 ms
epoch: 1 step: 5 | train loss: 4.040521 lr: 3.0000e-04 | took 1949.3284 ms
epoch: 1 step: 6 | train loss: 4.246991 lr: 3.0000e-04 | took 1965.3100 ms
epoch: 1 step: 7 | train loss: 3.724629 lr: 3.0000e-04 | took 1952.5358 ms
epoch: 1 step: 8 | train loss: 4.232645 lr: 3.0000e-04 | took 1998.7662 ms
epoch: 1 step: 9 | train loss: 4.279507 lr: 3.0000e-04 | took 2527.5224 ms
.. (truncated) ..
Running validation
val loss: 3.850747 | val_batches: 31 | validation took 18.2614 seconds
Model saved at model_checkpoints/gpt2.bin
.. (truncated) ..
Training Statistics
Best training loss: 3.138453
Best validation loss: 3.850747
Latest model checkpoint: model_checkpoints/gpt2.bin
```

</details>

Now, let's try inferring from the finetuned model on the same prompts as before.

```console
GPT2 Model Settings
+-----------------+----------------------------+
| Parameter       | Value                      |
+-----------------+----------------------------+
| max_block_size  | 1024                       |
| vocab_size      | 50257                      |
| n_layers        | 12                         |
| n_heads         | 12                         |
| n_embd          | 768                        |
| checkpoint_path | model_checkpoints/gpt2.bin |
| steps_trained   | 594                        |
+-----------------+----------------------------+

Starting Inference

:
Thou bless'st me, John.

<|endoftext|> feat:
Naugh, thou accompt'st by my merit,
her cogheth for trinum of wildliness; you blame
Her immoric architecture, not mine, mellows away
A hit too flat upon her swords while thou dost praise. Well,
I once saw her entrails on her back; take thee,
Whenime fares; and I would have told thee
To

Inference Time: 126.3122 seconds | tokens/s: 0.79
```

That's a lot better compared to the output that we got when we trained gpt2 from scratch. Switching to larger models often results in obtaining better results, however this requires longer time for training and inference.

### Autocompletion using Pretrained Models

Pretrained models are trained on a large dataset and we can leverage it to generate random stuff for us.

Example:

```bash
PROMPT="Support Vector Machines (SVM)"
TOKENS=$(python3 tokenizer.py -p "$PROMPT" -e)
OPENBLAS_NUM_THREADS=6 ./infer_gpt --load-checkpoint=model/gpt2-350M.bin --tokenizer=tokenizer.bin --prompt="$TOKENS" --max-tokens=100
```

&nbsp;
<p>"<i>Support Vector Machines (SVM) and many other mathematical techniques are available for students and researchers to explore in the lab. Though it isn't a formula that is easily understood by the layman there is a wealth of pre-programmed mathematical libraries available within the MathLab database to take advantage of.<|endoftext|>Find Great Equipment, Services & Featured Seller's TVs Included FREE2. Well, Start with When to Buy for TV Cables Price 49.30 € Buy Online $49.30 Roku 3C: 20m Stereo Xap
</i>"</p>
&nbsp;

Pretty cool! However, the model quickly loses the context and starts generating random things it learnt during training. It is understandable as these pretrained models are just document generators.

## Understanding gpt.c code

Since this project is for educational purposes, it is important to understand how layers are structured in gpt.c. Having this understanding will help you add your own layers to gpt.c or take ideas from this project and implement your own gpt.c in your own style! [Desigining a Layer](docs/design/layer.md) has detailed information on how layers are created in gpt.c. 

## Contributing Guidelines

I would like to keep this project as simple as possible. The main focus of this project is to create a GPT using C from scratch while keeping it simple, readable, easy to understand and have good performance at the same time. Feel free to contribute to this project and make it better and faster.

Your contributions can be one of, but not limited to:

- Improving code readability
- Code correctness
- Performance improvements without adding too much complexity
- Improving docs
- Design changes
- Improvements to CI/CD
- Improvements to Python scripts

## TODO

- [ ] Add support for CUDA
- [ ] Add support for MPS on Mac
- [ ] Add more optimizers
- [ ] Investigate performance issues in binaries generated by clang
- [ ] Create better test cases

## License<!-- omit from toc -->

MIT