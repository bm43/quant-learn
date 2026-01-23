import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ViTMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # common in ViT: one linear layer outputs q,k,v concatenated
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, dim) where N = num_patches (+1 if CLS token)
        mask (optional): broadcastable to (B, num_heads, N, N), True = masked
        """
        B, N, _ = x.shape

        qkv = self.qkv(x)  # (B, N, 3*dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, h, N, head_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, h, N, N) [web:173]

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, h, N, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, N, self.dim)  # (B, N, dim)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn
