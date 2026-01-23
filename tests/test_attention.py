import torch
import torch.nn as nn

from quant_learn.models.ml.transformer import multi_head_attention
from quant_learn.models.ml.vision_transformer import ViTMultiHeadSelfAttention
# test if there any shape issues

def test_mha():
    torch.manual_seed(0)

    B, N, d_k, h = 2, 11, 512, 8
    mha = multi_head_attention(d_k, h)

    x = torch.randn(B, N, d_k)

    # optional mask
    causal = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    mask = causal.unsqueeze(0).expand(B, -1, -1) # B x N x N

    y = mha(x, x, x, mask=mask)

    print("y shape: ", y.shape)

def test_vmha():
    torch.manual_seed(3)

    B = 2
    img_h = img_w = 32
    patch = 16
    dim = 64
    heads = 4

    num_patches = (img_h // patch) * (img_w // patch)  # 4 patches for 32x32 with 16x16 patches [web:166]
    N = num_patches + 1  # +1 for CLS token (common in ViT) [web:195]

    attn = ViTMultiHeadSelfAttention(dim=dim, num_heads=heads, attn_drop=0.0, proj_drop=0.0)

    # ViT block input is tokens, not images: (B, N, dim) [web:166]
    x = torch.randn(B, N, dim)

    y, A = attn(x)  # y: (B, N, dim), A: (B, heads, N, N)

    print(y.shape)

    assert y.shape == (B, N, dim)
    assert A.shape == (B, heads, N, N)  # attention over tokens (patches + cls) [web:196]
    assert torch.isfinite(y).all()
    assert torch.isfinite(A).all()

    # each row of attention should sum to ~1 because of softmax
    row_sums = A.sum(dim=-1)  # (B, heads, N)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_vmha()