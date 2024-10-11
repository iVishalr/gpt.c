import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import argparse


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
        print(f"Loading weight from pretrained gpt: {model_type}")

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
    print(f"Model saved at {os.path.join(dirname, filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="model.py",
        description="Reference GPT2 implementation using PyTorch."
    )
    parser.add_argument("--block-size", required=False, default=1024, help="Block size of the model. Default: 1024")
    parser.add_argument("--vocab-size", required=False, default=50257, help="Vocab size of the model. Default: 50257")
    parser.add_argument("--layers", required=False, default=12, help="Number of layers in the model. Default: 12")
    parser.add_argument("--heads", required=False, default=12, help="Number of heads in the model. Default: 12")
    parser.add_argument("--embd", required=False, default=768, help="Embedding dimension of the model. Default: 768")
    parser.add_argument("--from-pretrained", required=False, default=None, help="Pass name of model (gpt2, gpt2-medium, gpt2-large, gpt2-xl) to load pretrained weights. Default: None")
    parser.add_argument("--output", required=False, default="model", help="Pass path to directory to store model weights. Default: 'model/'")
    parser.add_argument("--name", required=False, default="gpt2", help="Pass a name for the model. Default: gpt2")

    args = parser.parse_args()

    if isinstance(args.from_pretrained, str):
        if args.from_pretrained in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]:
            model = GPT.from_pretrained(args.from_pretrained)
        else:
            raise ValueError(f"No pretrained weights available for {args.from_pretrained}.")
    elif args.from_pretrained is None:
        config = GPTConfig(
            block_size=int(args.block_size),
            vocab_size=int(args.vocab_size),
            n_head=int(args.heads),
            n_layer=int(args.layers),
            n_embd=int(args.embd)
        )
        model = GPT(config)
    else:
        raise ValueError(f"Expected a Optional[str] value for --from-pretrained. Got {type(args.from_pretrained).__name__}")
    
    print("Model configs:")
    print(model.config)

    print(f"Parameters: {sum([torch.numel(p) for p in model.parameters()]) / 1e6:.2f}M")

    model_save_path = os.path.join(args.output, f"{args.name}.bin")
    write_model(model, model_save_path, 0)