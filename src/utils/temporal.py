import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class FFN(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_in * 4)
        self.gelu = QuickGELU()
        self.fc2 = nn.Linear(dim_in * 4, dim_out)
    
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = int(dim / heads)
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, q, k, v, padding_mask, enable_proj: bool = True):
        nbatchs = q.size(0)
        if enable_proj is True:
            q = self.proj_q(q)
            k = self.proj_k(k)
            v = self.proj_v(v)
        q, k, v = [x.view(nbatchs, -1, self.heads, self.head_dim).transpose(1, 2) for x in (q, k, v)]
        out = self.attention(q, k, v, padding_mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(nbatchs, -1, self.head_dim * self.heads)
        if enable_proj is True:
            out = self.proj_out(out)
        return out
        
    def attention(self, q, k, v, padding_mask, dropout=None):
        batch, heads, length, dim = q.size()
        qk = torch.matmul(q, k.transpose(-2, -1)) / (dim ** 0.5)

        if padding_mask is not None:
            qk_padding_mask = padding_mask.unsqueeze(1).unsqueeze(1).repeat([1, heads, length, 1]).to(qk.device)
            qk = qk + qk_padding_mask

        qk_attn = F.softmax(qk, dim=-1)
        attn = F.softmax(qk_attn, dim=-1)
        attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)

        return torch.matmul(attn, v)

class TemporalTransformer(nn.Module):
    def __init__(self, dim, heads, dropout=0.):
        super().__init__()
        self.mha = MultiHeadAttention(dim, heads, dropout)
        self.ln_1 = LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim * 4, dim))
        ]))
        self.ln_2 = LayerNorm(dim)

    def forward(self, x: torch.Tensor, padding_mask = None):
        x = self.mha(x, x, x, padding_mask) + x
        x = self.ln_1(x)
        x = self.mlp(x) + x
        x = self.ln_2(x)
        return x

class DistanceAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__() 
        self.proj_in = nn.Linear(input_dim, input_dim)
        self.proj_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, padding_mask):
        b, t, d = x.size()
        x = self.proj_in(x)
        tmp_ones = torch.ones(t).to(x.device)
        tmp_n = torch.linspace(1, t, t).to(x.device)
        tg_tmp = torch.abs(tmp_n * tmp_ones - tmp_n.view(-1,1))
        tg_tmp = tg_tmp.unsqueeze(0).repeat([b, 1, 1])
        if padding_mask is not None:
            tg_tmp = -tg_tmp + padding_mask
        else:
            tg_tmp = -tg_tmp
        attn = torch.exp(tg_tmp / torch.exp(torch.tensor(1.)))

        return torch.matmul(attn, x)
    
class SimilarityAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__() 
        self.proj_in = nn.Linear(input_dim, input_dim)
        self.proj_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, padding_mask):
        b, t, d = x.size()
        x = self.proj_in(x)
        x = x / x.norm(dim=-1, keepdim=True)
        attn = torch.einsum('btd, btd -> btt', x, x)
        