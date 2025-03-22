# gpt.c<!-- omit from toc -->

<p align="center">
<img src="docs/images/banner.svg", alt="banner.svg"></img>
</p>

gpt.c is a simple C implementation of OpenAI's popular GPT-2 model. This project was originally inspired by Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c). However, llm.c has evolved into a great, but complex and slightly confusing codebase over time. This project aims to build GPT-2 from scratch in C, keeping it as simple as possible while still achieving very good performance, preferrably on par with Pytorch. To enhance understanding, most layers in gpt.c are designed in a way that makes building models very similar to doing so in PyTorch. 

## Table of Contents<!-- omit from toc -->

- [Clone Project](#clone-project)
- [Install dependencies](#install-dependencies)
- [Build gpt.c](#build-gptc)
- [Downloading datasets](#downloading-datasets)
- [Training](#training)
- [Inference](#inference)
- [Finetuning with Pretrained Weights](#finetuning-with-pretrained-weights)
- [Autocompletion using Pretrained Models](#autocompletion-using-pretrained-models)
- [Switching between C and Python](#switching-between-c-and-python)
- [Contributing Guidelines](#contributing-guidelines)
- [Need Help](#need-help)


## Clone Project

Clone the project along with submodules

```bash
git clone https://github.com/iVishalr/gpt.c.git
git submodule update --init --recursive
```

## Install dependencies

Currently, gpt.c runs on both Linux and macOS. gpt.c supports both cpu and cuda devices on Linux and cpu on macOS. 

**Linux**

```bash
sudo apt install libomp-dev build-essential gcc gfortran make valgrind
pip install -r requirements.txt
```

_Note: To use CUDA, please install nvcc from [here](https://developer.nvidia.com/cuda-downloads)._

**macOS**

```bash
brew install libomp
brew install argp-standalone
brew install gfortran
pip install -r requirements.txt
```

_Note: Requires `brew` package manager to be installed._

## Build gpt.c

gpt.c relies on OpenBLAS for accelerating matrix operations on CPU. OpenBLAS must be built before gpt.c

```bash
make third_party
```

Once the third party libraries are compiled, build gpt.c using 

```bash
make
```

Other make options include:

  - `CC`: Pass the compiler to use for building gpt.c (gcc | clang). Default: `gcc`.
  - `DEVICE`: Pass the device for which gpt.c should be built (cpu | cuda). Default: `cpu`.
  - `BUILD`: Pass the type of build to use (release | debug). Default: `release`.

_Note: Recommended compiler is `gcc`. `gcc` produces binaries that are 2x faster compared to `clang` (reason unknown). On macOS, when compiling using `gcc` compiler, the compilation happens using `clang` as the gcc binary is aliased to clang provided by Apple._

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

## Downloading datasets

The `data/` directory contains sample datasets that can be used with gpt.c. These python scripts downloads the data from internet and tokenizes using GPT-2 tokenizer and writes them into training (90%) and validation (10%) splits.

```bash
python3 data/tinyshakespeare.py
```

## Training

We need to create a initial model file using `model.py`. This model file will contain the initial untrained weights for the model that is organized in a way that gpt.c recognizes.

```bash
python3 model.py
```

This script creates a GPT2 model using PyTorch and dumps the model parameters into a .bin file. This file will be consumed by C executables for training and inference. 

By default, the above script creates the GPT2 124M parameter model with 12 layers, 12 heads, 1024 block size, 768 embedding dims and 50257 vocab size. You can override these as follows:

```bash
python3 model.py --block-size=128 --vocab-size=50257 --layers=2 --heads=12 --embd=128
```

_Note: Do not change the vocab size. GPT2 tokenizer uses a vocab size of 50257._

Start training the model by executing the following command:

```bash
OPENBLAS_NUM_THREADS=6 OPENMP_NUM_THREADS=6 ./train_gpt --train-data=data/tinyshakespeare/tinyshakespeare_train.bin --val-data=data/tinyshakespeare/tinyshakespeare_val.bin --max-epochs=5 --log-dir=model_checkpoints --output=gpt2 --batch-size=8 --block-size=128 --lr=3e-4 --val-block-size=128 --load-checkpoint=/home/vishalr/Desktop/gpt.c/model/gpt2.bin
```

To train using CUDA, use `--device=cuda` along with the above command.

_Note: It is highly recommended to set `OPENBLAS_NUM_THREADS` to the number of physical cores on your machine._

<details>

<summary>Training Output</summary>

```console
GPT2 Model Settings
+-----------------+--------------------------------------------+
| Parameter       | Value                                      |
+-----------------+--------------------------------------------+
| max_block_size  | 1024                                       |
| vocab_size      | 50257                                      |
| n_layers        | 12                                         |
| n_heads         | 12                                         |
| n_embd          | 768                                        |
| checkpoint_path | /home/vishalr/Desktop/gpt.c/model/gpt2.bin |
| steps_trained   | 0                                          |
+-----------------+--------------------------------------------+

Training Settings
+--------------------+------------------------------------------------+
| Parameter          | Value                                          |
+--------------------+------------------------------------------------+
| train_data         | data/tinyshakespeare/tinyshakespeare_train.bin |
| val_data           | data/tinyshakespeare/tinyshakespeare_val.bin   |
| log_dir            | model_checkpoints                              |
| save_checkpoint    | model_checkpoints/gpt2.bin                     |
| max_epochs         | 5                                              |
| train_batch_size   | 8                                              |
| train_block_size   | 128                                            |
| num_train_batches  | 297                                            |
| total_train_steps  | 1485                                           |
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
| device             | cuda                                           |
+--------------------+------------------------------------------------+

epoch: 1 step: 1 | train loss: 4.712266 lr: 3.0000e-04 | took 83.9127 ms
epoch: 1 step: 2 | train loss: 4.552876 lr: 3.0000e-04 | took 123.3569 ms
epoch: 1 step: 3 | train loss: 4.266821 lr: 3.0000e-04 | took 98.3756 ms
epoch: 1 step: 4 | train loss: 4.148801 lr: 3.0000e-04 | took 97.8375 ms
epoch: 1 step: 5 | train loss: 4.029813 lr: 3.0000e-04 | took 97.9755 ms
.. (truncated) ..
Running validation
val loss: 3.833156 | val_batches: 31 | validation took 1.2352 seconds
Model saved at model_checkpoints/gpt2.bin
.. (truncated) ..
Training Statistics
Best training loss: 0.891998
Best validation loss: 3.833156
Latest model checkpoint: model_checkpoints/gpt2.bin
```

</details>

## Inference

`infer_gpt` is used for perfoming inference in C. During inference, the user provides a prompt in text which gets tokenized into a list of int tokens which are then fed into the model. In each iteration, we perform a forward pass on the model to obtain logits. The next token is sampled from this logits and gets appended to the list of tokens which then gets used in the next forward pass step. 

Since gpt.c only understands tokens from GPT2 tokenizer, the user prompts must first be tokenized using the tokenizer to obtain the raw integer tokens. Additionally, the inference script also requires the tokenizer to be saved in a format that gpt.c understands. 

To save the GPT2 tokenizer, use the following:

```bash
python3 tokenizer.py -s
```

A `tokenizer.bin` file is created which contains the weights of the GPT2 tokenizer as obtained from tiktoken.

Now, we need to encode the user's prompts before passing to `infer_gpt`.

```bash
PROMPT="ANTONIO"
TOKENS=$(python3 tokenizer.py -p "$PROMPT" -e)
```

The generated tokens for the above PROMPT will look as follows:

```bash
$ echo $TOKENS
>>> [50256, 8643, 1340, 9399]
```

Now, we are ready to generate more tokens using `infer_gpt`.

```bash
OPENBLAS_NUM_THREADS=6 ./infer_gpt --load-checkpoint=model/gpt2.bin --tokenizer=tokenizer.bin --prompt="$TOKENS" --max-tokens=200 --interactive --device=cuda
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

<|endoftext|>ANTONIO:
The heavens see that we areget; and all good things
How we wish to musicians.
Let's be glad to share the mine, dear,
To have a curst gain over such means as
Look for or colds that ice-drops meet
On. But yet our consolation
Which wakes upon this part O' this more remains:
And so it might mark us to be neither great sooth.
Troth, to be ungrateful nature,
By oath double ingrate beasts.ross,
Gave way to the greater than that part,
Could then the weak first free of mortals
Unstuck; the ring had no more staying power
Than the bare upper self. The womb from the so-iceraved stom,
Slowed with the so Krafts to draw. Still had no more; it was but bootless back
This ribolith yielded more than a cracking forest: it had no more power
To

Inference Time: 15.1523 seconds | tokens/s: 13.20
```

</details>

## Finetuning with Pretrained Weights

Training a GPT2 from scratch may not be practical. To speed up training, you can download the pretrained weights for GPT2 and then fine tune on your dataset. 

In gpt.c, you can use the `model.py` script to download the pretrained weights from huggingface.

_Note: Caution, large files will be downloaded for gpt2-large and gpt2-xl._

```bash
python3 model.py --from-pretrained gpt2 --name gpt2
python3 model.py --from-pretrained gpt2-medium --name gpt2-medium
python3 model.py --from-pretrained gpt2-large --name gpt2-large
python3 model.py --from-pretrained gpt2-xl --name gpt2-xl
```

To finetune the 124M model (gpt2) by training it on our custom dataset, execute the following:

```bash
OPENBLAS_NUM_THREADS=6 ./train_gpt --train-data=data/tinyshakespeare/tinyshakespeare_train.bin --val-data=data/tinyshakespeare/tinyshakespeare_val.bin --max-epochs=5 --log-dir=model_checkpoints --output=gpt2 --batch-size=8 --block-size=128 --lr=3e-4 --load-checkpoint="./model/gpt2.bin" --device=cuda
```

Once training is complete, perform inference using the finetuned model to notice the improvements in text generation.

## Autocompletion using Pretrained Models

Pretrained models are trained on a large dataset and we can leverage it to generate random stuff for us.

Example:

```bash
PROMPT="Support Vector Machines (SVM)"
TOKENS=$(python3 tokenizer.py -p "$PROMPT" -e)
OPENBLAS_NUM_THREADS=6 ./infer_gpt --load-checkpoint=model/gpt2-medium.bin --tokenizer=tokenizer.bin --prompt="$TOKENS" --max-tokens=200 --device=cuda
```

```console
Support Vector Machines (SVM) and many other mathematical techniques are available for students and researchers to explore in the lab. Online resources include SVM instructor videos, videos on releasing SVM.<|endoftext|>Bargain Bay, the classic stadium featuring NHL legend Shayne Gostisbehere, could be reborn as part of a large redeveloped plaza that includes upscale condos, a grocery and dessert expo, parkland and a 400-space parking lot.

The complex not only awaits the government's approval to build suites in what would become Ontario Place, adjacent to City Hall, for the soccer team, but it has full "coexistence" with other projects and will deepen public understanding of the city, officials said Friday.

In the development, standard artificial turf will be mixed with the 'Jock Ideal' Ice, causing unique duck eggs to be hatched. Some partners, however, don't like that the goalposts would come off. It looks tragic if you have watched those pictures. — Robert Nechamelen
```

Pretty cool! However, the model quickly loses the context and starts generating random things that was learnt during training. It is understandable as these pretrained models are just document generators.

## Switching between C and Python

It is possible to use the model weights interchangeably with C and Python. The python training script `train_gpt.py` implements the training pipeline using PyTorch. Models trained using this script can later be used with `infer_gpt` C executable for performing inference. Similarly, models trained using `train_gpt` C executable can be loaded to the python script to continue training with PyTorch.

```console
$ python3 train_gpt.py -h
usage: train_gpt.py [-h] --train-data TRAIN_DATA [--val-data VAL_DATA] [--log-dir LOG_DIR] [--output OUTPUT] [--epochs EPOCHS] [--device DEVICE] [--batch-size BATCH_SIZE] [--block-size BLOCK_SIZE] [--lr LR] [--weight_decay WEIGHT_DECAY] [--torch-ckpt TORCH_CKPT] [--c-ckpt C_CKPT]

Trains GPT2 on a given training dataset.

options:
  -h, --help            show this help message and exit
  --train-data TRAIN_DATA
                        Path to training data generated by `prepro_xxx.py` script.
  --val-data VAL_DATA   Path to validation data generated by `prepro_xxx.py` script. Default: None
  --log-dir LOG_DIR     Log directory to store model checkpoints. Default: 'logs'
  --output OUTPUT       Name of the checkpoint to use for saving the model. Default: 'checkpoint'
  --epochs EPOCHS       Number of epochs to train model. Default: 10
  --device DEVICE       Device to train model on (cpu, cuda). Default: 'cpu'
  --batch-size BATCH_SIZE
                        Batch size to use for training GPT2. Default: 8
  --block-size BLOCK_SIZE
                        Block size to use for Dataloader for training GPT2. This option doesn't change model's block_size value. Default: 128
  --lr LR               Learning rate for training GPT2. Default: 3e-4
  --weight_decay WEIGHT_DECAY
                        Weight decay to use for training GPT2. Default: 0
  --torch-ckpt TORCH_CKPT
                        Path to torch checkpoint saved by torch.save(...). Default: None
  --c-ckpt C_CKPT       Path to C model checkpoint to load into torch model. Default: None

```

## Contributing Guidelines

Since this project is for educational purposes, it is important to understand how layers are structured in gpt.c. Having this understanding will help you add your own layers to gpt.c or take ideas from this project and implement your own gpt.c in your own style! [Desigining a Layer](docs/design/layer.md) has detailed information on how layers are created in gpt.c. 

I would like to keep this project as simple as possible. The main focus of this project is to create a GPT using C from scratch while keeping it simple, readable, easy to understand and have good performance at the same time. Feel free to contribute to this project and make it better and faster.

Your contributions can be one of, but not limited to:

- Improving code readability
- Code correctness
- Performance improvements without adding too much complexity
- Improving docs
- Design changes
- Improvements to CI/CD
- Improvements to Python scripts

## Need Help

- [ ] Add support for MPS on Mac
- [ ] Investigate issue with ioctl delays during training on CUDA. There's a 20ms delay caused when cudaStreamSynchronize() is called at the end of an iteration. Reducing this delay would make gpt.c match pytorch's training speeds (74ms).

## License<!-- omit from toc -->

MIT