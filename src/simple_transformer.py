import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class MultiHeadAttention(nn.Module):
    """支持 KV Cache 的多头自注意力层"""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除！"

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True, kv_cache=None):
        B, S, D = x.shape

        # 1. 投影 Q, K, V -> (B, num_heads, S, head_dim)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. KV Cache 核心逻辑：历史缓存拼接
        if kv_cache is not None:
            past_k, past_v = kv_cache
            # 在序列长度维度（dim=-2）进行拼接
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
        
        # 记录更新后的缓存，返回给下一步使用
        new_kv_cache = (k, v)

        # 3. 计算注意力: q * k^T
        # q 的长度是 S (decode 时为 1)，k 的长度是 total_seq_len (包含历史 token)
        total_s = k.shape[-2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 掩码：只有当 q 的长度 > 1 时（即批量 prefill 且没有 cache 时）才需要下三角因果掩码
        if is_causal and S > 1:
            mask = torch.full((S, total_s), float('-inf'), device=x.device, dtype=x.dtype)
            mask = torch.triu(mask, diagonal=1)
            scores = scores + mask

        # 5. 加权与输出
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(output), new_kv_cache


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 1024, num_heads: int = 16):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = SwiGLUMLP(dim, hidden_dim=int(dim * 2.67))

    def forward(self, x, is_causal: bool = True, kv_cache=None):
        norm_x = self.attn_norm(x)
        attn_out, new_kv_cache = self.attn(norm_x, is_causal=is_causal, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_kv_cache