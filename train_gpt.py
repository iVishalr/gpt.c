"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import os
import math
import sys
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CausalSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", 
            torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1,1, config.block_size, config.block_size)
        )        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch_size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, value for all heads and move the head dimension forward
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = -1)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, nh, T, hs)

        # if args.flash_attention:
        #     y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # else:
        #    manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, return_logits = True):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device) # (t,)

        # forward the GPT model itself
        tok_embd = self.transformer.wte(idx)
        pos_embd = self.transformer.wpe(pos)
        x = tok_embd + pos_embd

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model weights from huggingface."""

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weight from pretrained gpt: {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only rhe top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim = -1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

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


def write_model(model: GPT, filename):
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
        print(_shape)
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

        print("[GPT]")
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

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
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
    ckpt_path: str = "./transformers.pt"

class Trainer:
    def __init__(self, model: GPT, configs: TrainingConfig, train_set: str, test_set: Optional[str] = None) -> None:
        self.model = model
        self.configs = configs
        self.train_set = train_set
        self.test_set = test_set

        self.device = torch.device("cpu")

        if self.configs.device == "cuda" and torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = nn.DataParallel(self.model).to(self.device)
        
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        print(f"saving model at {self.configs.ckpt_path}.")
        torch.save(raw_model.state_dict(), self.configs.ckpt_path)

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
            if self.configs.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

if __name__ == "__main__":

    config = GPTConfig()
    model = GPT(config)
    load_model(model, "./model/gpt2.bin")
    # write_model(model, "model/gpt2.bin")

    training_configs = TrainingConfig(eps=1e-8)

    trainer = Trainer(
        model = model,
        configs=training_configs,
        train_set="./data/tiny_shakespeare/tiny_shakespeare_train.bin"
    )

    trainer.train()
