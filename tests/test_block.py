import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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

blk_attn_qkv_weight = [
  [ 1.117950, 0.308147, 1.014249, 1.042115, 1.247756, -0.049397 ],
  [ 0.200683, 0.987238, 0.096329, 0.598036, 0.458942, 0.734093 ],
  [ 0.254381, 0.524343, 1.321474, 1.256017, 0.746520, 0.894719 ],
  [ -0.151028, 0.694309, -0.378638, 0.032955, -0.158968, 1.052536 ],
  [ -0.123641, 0.320066, -0.172484, -0.210597, 1.406295, -0.011785 ],
  [ 0.523492, 1.115996, 0.704610, 0.129492, 0.749863, 0.544118 ],
  [ 0.488344, 1.358794, 0.123107, 0.992920, 0.548582, 0.990298 ],
  [ 0.318766, 1.211212, 0.106392, 0.231991, 1.058980, 1.261160 ],
  [ -0.281538, 1.316201, 0.547221, -0.251928, -0.059092, 0.796501 ],
  [ 1.208856, 0.225515, -0.291681, -0.371876, 0.423165, -0.293635 ],
  [ 0.024586, 1.354905, 1.230610, 1.137444, 0.076149, 0.572225 ],
  [ 0.273314, 0.972741, 0.522771, 0.804670, 0.557413, -0.336896 ],
  [ 0.386719, 1.284427, 1.282564, 0.901359, 0.108170, 0.933297 ],
  [ 0.754271, 0.234880, 0.841250, -0.106757, 0.391200, 1.190405 ],
  [ 1.097993, 0.191808, 0.007672, 1.214560, 0.228180, 0.839085 ],
  [ 1.329173, 0.661014, 0.785742, 1.151534, 0.390211, 1.270140 ],
  [ 0.315511, 1.071773, 0.834632, 1.246529, 0.468194, -0.016203 ],
  [ 1.317882, 1.263161, -0.140024, 1.192198, 0.756272, 0.376394 ],
]

blk_attn_qkv_bias = [ 0.717247, 0.102295, 1.019522, 0.150248, 0.403787, 0.002474, -0.067595, 0.093531, 0.602530, 0.348325, -0.100158, 1.238958, -0.220838, -0.179233, 0.491724, 0.973152, 1.380550, 1.290183 ]

blk_attn_c_proj_weight = [
  [ 0.835044, 0.287812, 0.953708, 0.261428, 0.126093, 0.013654 ],
  [ 0.653473, 0.035727, -0.131433, 0.921697, -0.180324, 1.033088 ],
  [ -0.110158, 0.945171, -0.272865, 1.317612, -0.312829, 0.539170 ],
  [ -0.088162, 0.027824, 1.040949, 0.922616, 0.784397, 1.349040 ],
  [ 0.753326, 0.971807, -0.238441, -0.163198, 0.536712, -0.266140 ],
  [ -0.281264, -0.036493, 0.429920, 1.080693, 0.633183, 0.964262 ],
]

blk_attn_c_proj_bias = [ -0.313902, -0.121592, 1.408237, -0.037086, 1.208353, -0.180335 ]

blk_ln1_weight = [ 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 ]

blk_ln1_bias = [ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 ]

blk_ln2_weight = [ 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000 ]

blk_ln2_bias = [ 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000 ]

blk_mlp_c_fc_weight = [
  [ 1.404250, -0.310053, 1.173084, -0.276863, -0.400689, 1.268504 ],
  [ 0.670555, -0.080603, -0.111920, 0.303256, 1.250262, 1.080725 ],
  [ 0.244047, 0.595339, 0.644284, 0.413854, 0.840389, -0.227252 ],
  [ 0.555963, 0.967373, 0.144503, 1.394131, 0.639818, 1.185934 ],
  [ 0.950145, 0.734164, -0.343906, 0.950133, 1.105327, 1.272695 ],
  [ 1.178046, 1.101329, 1.370890, 0.942882, 1.232713, 1.378450 ],
  [ 0.803137, 0.495020, -0.110401, 1.099465, 1.206524, -0.268388 ],
  [ 0.771942, 0.042323, 0.735199, 0.007978, 0.864425, 0.167340 ],
  [ 0.188974, 0.012140, -0.273535, 0.741725, -0.001977, 0.774531 ],
  [ 0.519411, 1.356416, 0.100447, 0.583753, 0.898301, -0.202475 ],
  [ 0.448200, 0.668098, 1.307102, 0.410842, 0.202732, 1.131567 ],
  [ 0.381044, -0.402378, 0.218339, 0.678891, 1.105335, 0.016615 ],
  [ 0.818752, 0.469029, 0.467187, 0.145703, 0.885256, -0.076636 ],
  [ 0.721291, -0.334018, 0.343752, 0.856004, 0.815955, 0.750023 ],
  [ 0.222287, -0.072882, 0.698191, 0.730982, 0.919119, 0.188243 ],
  [ 0.936756, -0.040929, 1.264590, 0.835610, 0.778161, 0.059074 ],
  [ 0.558929, -0.249044, 0.064944, 1.185517, 0.838095, -0.237969 ],
  [ -0.206116, 0.248599, 0.639308, 0.669318, 0.802549, 0.116316 ],
  [ 1.000930, 0.115592, 0.190545, -0.063566, 1.379844, -0.401748 ],
  [ 1.094705, 0.193882, -0.066382, 0.384647, 1.333112, 1.260986 ],
  [ 0.981138, 0.861620, -0.188192, 0.837479, 0.288981, 0.998217 ],
  [ 1.304801, 1.256158, 1.157421, -0.038504, 1.033427, 0.587268 ],
  [ 0.131775, 1.235559, 1.244115, 1.179332, 0.496628, 0.638416 ],
  [ -0.112601, 0.089310, 1.162256, 0.486193, 0.433992, 1.133852 ],
]

blk_mlp_c_fc_bias = [ 0.492693, 0.120449, -0.080514, 0.834559, 0.913344, -0.155650, 0.687297, 0.486234, 1.114218, 0.907353, -0.084535, -0.005049, 0.497322, -0.187982, -0.157139, 0.246495, 0.181763, 1.284536, 1.242011, 0.721786, 1.111846, 1.077878, 0.492869, 0.200227 ]

blk_mlp_c_proj_weight = [
  [ 0.351186, 0.723674, 0.653334, 0.160480, 0.008845, -0.101963, -0.052247, 0.707304, 0.307912, 0.201831, 0.262549, -0.075764, 0.397659, 1.111876, 0.617688, 0.169711, 0.723555, 0.868648, 0.482294, 0.017354, 1.039411, 0.676968, 0.524947, 0.088571 ],
  [ 0.581060, 0.396070, 0.964636, 0.351273, 0.139949, 0.254984, 0.822996, 0.695259, 1.182782, 0.272206, 1.059863, -0.012497, 0.374367, -0.196508, 0.898931, 0.886404, 0.209448, -0.042643, 1.014764, 0.811231, -0.134891, 0.428327, 1.185066, 0.792788 ],
  [ 0.092851, 0.463236, 1.014266, -0.071862, -0.063920, 0.335088, 0.220833, 0.721264, 0.935283, -0.018655, -0.131588, -0.128892, 0.440453, 0.895532, 0.770491, 0.419111, -0.036386, 0.626231, 0.610737, 0.542106, 0.633847, 0.305545, 0.224385, 1.047419 ],
  [ 0.467025, 0.035025, 0.654526, 0.536258, 0.667476, 0.635468, 0.124922, 0.964452, -0.105420, -0.064937, 1.096714, 0.034784, 0.474276, 0.113423, 0.960172, 0.205435, 0.298892, 1.032709, 0.280667, 0.943470, 0.724117, -0.152966, 0.158456, 0.891855 ],
  [ 0.677389, 0.973318, 0.229837, 0.107112, 0.074738, 0.658346, -0.049593, 0.745888, 0.897495, 0.809057, 0.078022, 0.360848, 0.240401, 0.407068, 0.121175, 0.339105, 0.546255, 0.013765, 0.578014, -0.183593, 0.331313, 0.334062, 0.225966, 0.834329 ],
  [ 0.162647, 0.710757, 0.573674, 1.090888, 0.761915, 0.936255, 0.778619, 0.235180, 0.705448, -0.195668, 0.546416, 0.984310, 0.666802, 0.700946, 0.526073, 0.360174, 0.305879, 0.808219, 0.925146, 0.750404, 0.011163, -0.157803, -0.114615, 0.761542 ],
]

blk_mlp_c_proj_bias = [ 0.060086, 0.667523, 0.782074, 0.595523, -0.202539, -0.196084 ]

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

class MLP(nn.Module):

    def __init__(self, n_embd, expansion_factor = 4):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, expansion_factor * n_embd)
        self.gelu    = nn.GELU('tanh')
        self.c_proj  = nn.Linear(expansion_factor * n_embd, n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, bias = True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias = bias)
        self.attn = SelfAttention(n_embd, n_heads, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, 4)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

X = torch.tensor(X, requires_grad=True)

blk_ln1_weight = nn.Parameter(torch.tensor(blk_ln1_weight))
blk_ln1_bias = nn.Parameter(torch.tensor(blk_ln1_bias))
blk_ln2_weight = nn.Parameter(torch.tensor(blk_ln2_weight))
blk_ln2_bias = nn.Parameter(torch.tensor(blk_ln2_bias))

blk_attn_qkv_weight = nn.Parameter(torch.tensor(blk_attn_qkv_weight))
blk_attn_qkv_bias = nn.Parameter(torch.tensor(blk_attn_qkv_bias))
blk_attn_c_proj_weight = nn.Parameter(torch.tensor(blk_attn_c_proj_weight))
blk_attn_c_proj_bias = nn.Parameter(torch.tensor(blk_attn_c_proj_bias))

blk_mlp_c_fc_weight = nn.Parameter(torch.tensor(blk_mlp_c_fc_weight))
blk_mlp_c_fc_bias = nn.Parameter(torch.tensor(blk_mlp_c_fc_bias))
blk_mlp_c_proj_weight = nn.Parameter(torch.tensor(blk_mlp_c_proj_weight))
blk_mlp_c_proj_bias = nn.Parameter(torch.tensor(blk_mlp_c_proj_bias))

blk = Block(6, 2, 2)

blk.ln_1.weight = blk_ln1_weight
blk.ln_1.bias = blk_ln1_bias
blk.ln_2.weight = blk_ln2_weight
blk.ln_2.bias = blk_ln2_bias

blk.attn.qkv.weight = blk_attn_qkv_weight
blk.attn.qkv.bias = blk_attn_qkv_bias
blk.attn.c_proj.weight = blk_attn_c_proj_weight
blk.attn.c_proj.bias = blk_attn_c_proj_bias

blk.mlp.c_fc.weight = blk_mlp_c_fc_weight
blk.mlp.c_fc.bias = blk_mlp_c_fc_bias
blk.mlp.c_proj.weight = blk_mlp_c_proj_weight
blk.mlp.c_proj.bias = blk_mlp_c_proj_bias

out = blk(X)
out.backward(torch.ones_like(out))

print(f"{X.grad = }")
print(f"{X.grad.shape = }")

print(f"{blk.attn.qkv.weight.grad = }")
print(f"{blk.attn.qkv.weight.grad.shape = }")
print(f"{blk.attn.qkv.bias.grad = }")
print(f"{blk.attn.qkv.bias.grad.shape = }")

print(f"{blk.attn.c_proj.weight.grad = }")
print(f"{blk.attn.c_proj.weight.grad.shape = }")
print(f"{blk.attn.c_proj.bias.grad = }")
print(f"{blk.attn.c_proj.bias.grad.shape = }")

print(f"{blk.ln_1.weight.grad = }")
print(f"{blk.ln_1.weight.grad.shape = }")
print(f"{blk.ln_1.bias.grad = }")
print(f"{blk.ln_1.bias.grad.shape = }")

print(f"{blk.ln_2.weight.grad = }")
print(f"{blk.ln_2.weight.grad.shape = }")
print(f"{blk.ln_2.bias.grad = }")
print(f"{blk.ln_2.bias.grad.shape = }")

print(f"{blk.mlp.c_fc.weight.grad = }")
print(f"{blk.mlp.c_fc.weight.grad.shape = }")
print(f"{blk.mlp.c_fc.bias.grad = }")
print(f"{blk.mlp.c_fc.bias.grad.shape = }")

print(f"{blk.mlp.c_proj.weight.grad = }")
print(f"{blk.mlp.c_proj.weight.grad.shape = }")
print(f"{blk.mlp.c_proj.bias.grad = }")
print(f"{blk.mlp.c_proj.bias.grad.shape = }")
