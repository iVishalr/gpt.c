import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

X = [
  [
    [ -0.243308, -0.901952, 1.517912, -0.504347, 0.724619, 0.078169 ],
    [ -1.037727, 1.057526, 0.831446, 0.505472, 0.966881, 0.274284 ],
  ],
  [
    [ -0.429154, -0.187968, 1.155797, 0.308952, 0.577628, -1.366481 ],
    [ 0.102991, 1.425297, 0.904628, 0.905223, 0.507633, -0.371706 ],
  ],
  [
    [ 0.515019, -0.240407, -0.172645, 0.574103, -1.252985, 0.368966 ],
    [ 0.380364, 0.080606, -1.618588, 0.370919, -0.809955, 1.243616 ],
  ],
  [
    [ -0.456103, 1.441584, -0.452529, 0.199174, -0.706242, -0.147632 ],
    [ -0.392105, -0.100300, 0.601288, -1.031887, 0.211699, -0.157713 ],
  ],
]

qkv_weight = [
  [ 1.117950, 0.308147, 1.014249, 1.042115, 1.247756, -0.049397 ],
  [ 0.200683, 0.987238, 0.096329, 0.598036, 0.458942, 0.734094 ],
  [ 0.254382, 0.524343, 1.321474, 1.256017, 0.746520, 0.894719 ],
  [ -0.151028, 0.694309, -0.378638, 0.032955, -0.158968, 1.052536 ],
  [ -0.123641, 0.320066, -0.172484, -0.210597, 1.406295, -0.011785 ],
  [ 0.523492, 1.115996, 0.704610, 0.129492, 0.749863, 0.544118 ],
  [ 0.488344, 1.358794, 0.123108, 0.992920, 0.548582, 0.990298 ],
  [ 0.318766, 1.211212, 0.106392, 0.231991, 1.058981, 1.261160 ],
  [ -0.281538, 1.316201, 0.547221, -0.251928, -0.059092, 0.796501 ],
  [ 1.208856, 0.225515, -0.291681, -0.371876, 0.423165, -0.293635 ],
  [ 0.024586, 1.354905, 1.230610, 1.137445, 0.076149, 0.572225 ],
  [ 0.273314, 0.972741, 0.522771, 0.804670, 0.557413, -0.336896 ],
  [ 0.386719, 1.284427, 1.282565, 0.901359, 0.108170, 0.933297 ],
  [ 0.754271, 0.234880, 0.841250, -0.106757, 0.391200, 1.190405 ],
  [ 1.097993, 0.191808, 0.007672, 1.214560, 0.228180, 0.839085 ],
  [ 1.329173, 0.661015, 0.785742, 1.151534, 0.390211, 1.270140 ],
  [ 0.315511, 1.071773, 0.834632, 1.246529, 0.468194, -0.016203 ],
  [ 1.317882, 1.263162, -0.140024, 1.192198, 0.756272, 0.376394 ],
]

qkv_bias = [ 0.717247, 0.102295, 1.019522, 0.150248, 0.403787, 0.002474, -0.067595, 0.093531, 0.602530, 0.348325, -0.100158, 1.238958, -0.220838, -0.179233, 0.491724, 0.973152, 1.380550, 1.290183 ]

c_proj_weight = [
  [ 0.835044, 0.287812, 0.953708, 0.261428, 0.126093, 0.013654 ],
  [ 0.653473, 0.035727, -0.131433, 0.921697, -0.180324, 1.033088 ],
  [ -0.110158, 0.945171, -0.272865, 1.317613, -0.312829, 0.539170 ],
  [ -0.088162, 0.027824, 1.040949, 0.922616, 0.784397, 1.349040 ],
  [ 0.753326, 0.971807, -0.238441, -0.163198, 0.536712, -0.266140 ],
  [ -0.281264, -0.036493, 0.429920, 1.080693, 0.633183, 0.964262 ],
]

c_proj_bias = [ -0.313902, -0.121592, 1.408237, -0.037086, 1.208353, -0.180335 ]

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, block_size) -> None:
        super().__init__()

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.block_size = block_size

        self.qkv = nn.Linear(n_embd, n_embd * 3)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim = -1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

attn = SelfAttention(6, 2, 2)

qkv_weight = nn.Parameter(torch.tensor(qkv_weight))
qkv_bias = nn.Parameter(torch.tensor(qkv_bias))

c_proj_weight = nn.Parameter(torch.tensor(c_proj_weight))
c_proj_bias = nn.Parameter(torch.tensor(c_proj_bias))

attn.qkv.weight = qkv_weight
attn.qkv.bias = qkv_bias

attn.c_proj.weight = c_proj_weight
attn.c_proj.bias = c_proj_bias

X = torch.tensor(X, requires_grad = True)

s = time.time()
out = attn(X)
print(f"Forward Pass took {time.time() - s:.6f} seconds")

print(f"{out = }", out.shape)

out.backward(torch.ones_like(out))

print(f"{attn.qkv.weight.grad = }")
print(f"{attn.qkv.bias.grad = }")

print(f"{attn.c_proj.weight.grad = }")
print(f"{attn.c_proj.bias.grad = }")

print(f"{X.grad = }")