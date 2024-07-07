"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataclasses import dataclass
from typing import Tuple, Optional, Generator
from tqdm import tqdm

from model import GPT, GPTConfig

def write_fp32(tensor, file):
    if tensor is None:
        return
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file):
    write_fp32(model_tensors["transformer.wte.weight"], file) # (vocab_size, C)
    write_fp32(model_tensors["transformer.wpe.weight"], file) # (block_size, C)
    
    # write block's parameters
    for i in range(L):
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
        write_fp32(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fp32(model_tensors["transformer.ln_f.weight"], file)
    write_fp32(model_tensors["transformer.ln_f.bias"], file)


def write_model(model: GPT, filename: str, step: int = 0):
    dirname, filename = os.path.dirname(filename), os.path.basename(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240415
    header[1] = model.config.block_size
    header[2] = model.config.vocab_size
    header[3] = model.config.n_layer
    header[4] = model.config.n_head
    header[5] = model.config.n_embd

    params = {name: param.cpu() for name, param in model.named_parameters()}
    num_required_shape_headers = 0
    shapes = []
    for name, param in params.items():
        _shape = list(param.shape)
        num_required_shape_headers += len(_shape) + 1
        shapes.append(_shape)
    
    shape_headers = torch.zeros(num_required_shape_headers, dtype=torch.int32)
    shape_headers_index = 0
    for i in range(len(shapes)):
        shape_headers[shape_headers_index] = len(shapes[i])
        shape_headers_index += 1
        for j in shapes[i]:
            shape_headers[shape_headers_index] = j
            shape_headers_index += 1

    header[6] = num_required_shape_headers
    header[7] = step

    with open(os.path.join(dirname, filename), "wb") as file:
        file.write(header.numpy().tobytes())
        file.write(shape_headers.numpy().tobytes())
        write_tensors(params, model.config.n_layer, file)
    print(f"Model saved at {filename}")

def load_model(model: GPT, filename: str) -> None:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"No such file or directory. {filename}")

    print(f"Loading checkpoint from {filename}")

    expected_magic_number = 20240415
    with open(filename, "rb") as file:  
        # read headers
        data = file.read(256 * np.dtype(np.int32).itemsize)
        data_np = np.frombuffer(data, dtype=np.int32)
        headers = np.copy(data_np) # we need to do this to avoid "The given NumPy array is not writable, and PyTorch does not support non-writable tensors" warning
        # validate headers
        if headers[0] != expected_magic_number:
            raise ValueError(f"Expected magic number in model to be {expected_magic_number}. Got {headers[0]}.")
        if headers[1] != model.config.block_size:
            raise ValueError(f"Expected block_size in checkpoint to be {model.config.block_size}. Got {headers[1]}.")
        if headers[2] != model.config.vocab_size:
            raise ValueError(f"Expected vocab_size in checkpoint to be {model.config.vocab_size}. Got {headers[2]}.")
        if headers[3] != model.config.n_layer:
            raise ValueError(f"Expected n_layer in checkpoint to be {model.config.n_layer}. Got {headers[3]}.")
        if headers[4] != model.config.n_head:
            raise ValueError(f"Expected n_head in checkpoint to be {model.config.n_head}. Got {headers[4]}.")
        if headers[5] != model.config.n_embd:
            raise ValueError(f"Expected n_embd in checkpoint to be {model.config.n_embd}. Got {headers[5]}.")
        
        params = {name: param.cpu() for name, param in model.named_parameters()}
        state_dict = model.state_dict()

        num_required_shape_headers = 0
        for name, param in params.items():
            _shape = list(param.shape)
            num_required_shape_headers += len(_shape) + 1

        if headers[6] != num_required_shape_headers:
            raise ValueError(f"Expected shape_headers in checkpoint to be {num_required_shape_headers}. Got {headers[6]}.")

        print(f"[GPT2 | steps trained: {headers[7]}]")
        print(f"max_block_size: {headers[1]}")
        print(f"vocab_size: {headers[2]}")
        print(f"n_layers: {headers[3]}")
        print(f"n_heads: {headers[4]}")
        print(f"n_embd: {headers[5]}")
        
        data = file.read(num_required_shape_headers * np.dtype(np.int32).itemsize)
        data_np = np.frombuffer(data, dtype=np.int32)
        data = np.copy(data_np).tolist()

        shapes = []
        idx = 0
        while idx < len(data):
            ndims = data[idx]
            shapes.append(tuple(data[idx + 1: idx + 1 + ndims]))
            idx += ndims + 1

        loaded_parameters = []
        for shape in shapes:
            numel = 1
            for dim in shape:
                numel *= dim
            data = file.read(numel * np.dtype(np.float32).itemsize)
            data_np = np.frombuffer(data, dtype=np.float32)
            data_np = np.copy(data_np)
            tensor = torch.from_numpy(data_np).view(shape)
            loaded_parameters.append(tensor)

        params["transformer.wte.weight"] = loaded_parameters[0] # (vocab_size, C)
        params["transformer.wpe.weight"] = loaded_parameters[1] # (block_size, C)

        # load block's parameters
        idx = 2
        for i in range(model.config.n_layer):
            params[f"transformer.h.{i}.ln_1.weight"] = loaded_parameters[idx]
            params[f"transformer.h.{i}.ln_1.bias"] = loaded_parameters[idx + 1]
            params[f"transformer.h.{i}.attn.c_attn.weight"] = loaded_parameters[idx + 2]
            params[f"transformer.h.{i}.attn.c_attn.bias"] = loaded_parameters[idx + 3]
            params[f"transformer.h.{i}.attn.c_proj.weight"] = loaded_parameters[idx + 4]
            params[f"transformer.h.{i}.attn.c_proj.bias"] = loaded_parameters[idx + 5]
            params[f"transformer.h.{i}.ln_2.weight"] = loaded_parameters[idx + 6]
            params[f"transformer.h.{i}.ln_2.bias"] = loaded_parameters[idx + 7]
            params[f"transformer.h.{i}.mlp.c_fc.weight"] = loaded_parameters[idx + 8]
            params[f"transformer.h.{i}.mlp.c_fc.bias"] = loaded_parameters[idx + 9]
            params[f"transformer.h.{i}.mlp.c_proj.weight"] = loaded_parameters[idx + 10]
            params[f"transformer.h.{i}.mlp.c_proj.bias"] = loaded_parameters[idx + 11]
            idx += 12

        params["transformer.ln_f.weight"] = loaded_parameters[idx]
        params["transformer.ln_f.bias"] = loaded_parameters[idx + 1]
        idx += 2

        params['lm_head.weight'] = params['transformer.wte.weight']

        # merge actual state_dict with params
        for k in state_dict:
            if k not in params: continue
            state_dict[k] = params[k]

        model.load_state_dict(state_dict)
        return headers[7]

class DataLoader:
    def __init__(self, input_path: str, batch_size: int = 8, block_size: int = 64):
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No such file or directory. {input_path}")

        self.input_path = input_path
        self.batch_size = batch_size
        self.block_size = block_size

        tokens = None
        with open(self.input_path, "rb") as f:
            tokens = np.frombuffer(f.read(), dtype=np.int32)
        
        self.data = torch.tensor(tokens, dtype=torch.long)
        self._current_pos = 0

    def __len__(self) -> int:
        return len(self.data) // (self.batch_size * self.block_size)

    def __iter__(self) -> torch.Tensor:
        return self.__next__()

    def __next__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        assert self.batch_size * self.block_size + 1 <= len(self.data), f"Not enough tokens found in {self.input_path} for batch_size={self.batch_size} and block_size={self.block_size}"
        batch = 0
        while batch < len(self):
            i = self._current_pos
            x = self.data[i : i + self.batch_size * self.block_size].view(self.batch_size, self.block_size)
            y = self.data[i+1:i+self.batch_size * self.block_size + 1].view(self.batch_size, self.block_size)
            self._current_pos += self.batch_size * self.block_size
            if self._current_pos + self.batch_size * self.block_size + 1 >= len(self.data):
                self._current_pos = 0
            batch += 1
            yield x, y

@dataclass
class TrainingConfig:
    max_epochs: int = 100
    block_size: int = 128
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9,0.999)
    weight_decay: float = 0.0
    eps: float = 10e-8
    grad_norm_clip = 1.0
    batch_size: float = 8
    device: str = "cpu"
    torch_ckpt_path: str = "transformers.pt"
    c_ckpt_path: str = "transformer.bin"

class Trainer:
    def __init__(self, model: GPT, configs: TrainingConfig, train_set: str, test_set: Optional[str] = None) -> None:
        self.model = model
        self.configs = configs
        self.train_set = train_set
        self.test_set = test_set
        self.steps = 0

        self.device = torch.device("cpu")

        if self.configs.device == "cuda" and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = nn.DataParallel(self.model).to(self.device)
        
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"Saving torch model at {self.configs.torch_ckpt_path}.")
        torch.save(raw_model.state_dict(), self.configs.torch_ckpt_path)
        print(f"Saving C model at {self.configs.c_ckpt_path}.")
        write_model(raw_model, self.configs.c_ckpt_path, step=self.steps)


    def train(self):
        model, config = self.model, self.configs
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        lr = config.lr
        optimizer = optim.AdamW(model.parameters(), lr, config.betas, config.eps, config.weight_decay)

        def run_epoch(split):
            is_train = split=="train"
            if is_train:
                model.train()
            else:
                model.eval()
            
            data = self.train_set if is_train else self.test_set
            loader = DataLoader(data, config.batch_size, config.block_size)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            losses = []

            for ix, data in pbar:
                x, y = data
                x = x.to(self.device)
                y = y.to(self.device)

                logits, loss = model(x,y)
                loss = loss.mean()
                losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    pbar.set_description(f"epoch {epoch+1} it: {ix+1} | loss: {loss.item():.5f} lr: {lr:e}")
                    self.steps += 1
               
                if not is_train:
                    test_loss = float(np.mean(losses))
                    print("test loss : ", test_loss)
                    return test_loss

        best_loss = float('inf')
        test_loss = float('inf')
        self.tokens = 0
        for epoch in range(self.configs.max_epochs):
            run_epoch('train')

            if self.test_set is not None:
                test_loss = run_epoch('test')
            
            good_model = self.test_set is None or test_loss < best_loss
            if (self.configs.torch_ckpt_path is not None or self.configs.c_ckpt_path is not None) and good_model:
                best_loss = test_loss
                self.save_checkpoint()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="train_gpt.py",
        description="Trains GPT2 on a given training dataset."
    )
    parser.add_argument("--train-data", type=str, required=True, help="Path to training data generated by `prepro_xxx.py` script.")
    parser.add_argument("--val-data", type=str, default=None, required=False, help="Path to validation data generated by `prepro_xxx.py` script. Default: None")
    parser.add_argument("--log-dir", type=str, default="logs", required=False, help="Log directory to store model checkpoints. Default: 'logs'")
    parser.add_argument("--output", type=str, default="checkpoint", required=False, help="Name of the checkpoint to use for saving the model. Default: 'checkpoint'")
    parser.add_argument("--epochs", type=int, default=10, required=False, help="Number of epochs to train model. Default: 10")
    parser.add_argument("--device", type=str, default="cpu", required=False, help="Device to train model on (cpu, cuda). Default: 'cpu'")
    parser.add_argument("--batch-size", type=int, default=8, required=False, help="Batch size to use for training GPT2. Default: 8")
    parser.add_argument("--block-size", type=int, default=128, required=False, help="Block size to use for Dataloader for training GPT2. This option doesn't change model's block_size value. Default: 128")
    parser.add_argument("--lr", type=float, default=3e-4, required=False, help="Learning rate for training GPT2. Default: 3e-4 ")
    parser.add_argument("--weight_decay", type=float, default=0, required=False, help="Weight decay to use for training GPT2. Default: 0")
    parser.add_argument("--torch-ckpt", type=str, default=None, required=False, help="Path to torch checkpoint saved by torch.save(...). Default: None")
    parser.add_argument("--c-ckpt", type=str, default=None, required=False, help="Path to C model checkpoint to load into torch model. Default: None")

    args = parser.parse_args()

    train_data_path = args.train_data
    val_data_path = args.val_data
    log_dir = args.log_dir
    output = args.output
    device = args.device
    max_epochs = args.epochs
    batch_size = args.batch_size
    block_size = args.block_size
    lr = args.lr
    weight_decay = args.weight_decay
    torch_ckpt = args.torch_ckpt
    c_ckpt = args.c_ckpt

    if torch_ckpt and c_ckpt:
        raise ValueError(f"Provide either --torch-ckpt or --c-ckpt flags but not both at the same time.")

    if not os.path.exists(log_dir):
        print(f"Creating {log_dir}")
        os.makedirs(log_dir)

    torch_ckpt_path = os.path.join(log_dir, f'{args.output}.pt')
    cmodel_ckpt_path = os.path.join(log_dir, f'{args.output}.bin')

    config = GPTConfig()
    model = GPT(config)
    
    steps_trained = 0
    if torch_ckpt:
        model.load_state_dict(torch.load(torch_ckpt, map_location=torch.device(device)), strict=True)
    elif c_ckpt:
        steps_trained = load_model(model, c_ckpt)

    training_configs = TrainingConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        block_size=block_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        torch_ckpt_path=torch_ckpt_path,
        c_ckpt_path=cmodel_ckpt_path,
    )

    trainer = Trainer(
        model = model,
        configs=training_configs,
        train_set=train_data_path,
        test_set=val_data_path
    )
    trainer.steps = steps_trained
    trainer.train()
